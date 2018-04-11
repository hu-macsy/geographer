/*
 * SpectralPartition
 *
 *  Created on: 15.03.17
 *      Author: tzovas
 */


#include "SpectralPartition.h"
#include "GraphUtils.h"

using scai::lama::Scalar;

namespace ITI {


template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> SpectralPartition<IndexType, ValueType>::getPartition(const CSRSparseMatrix<ValueType> &adjM, const std::vector<DenseVector<ValueType>> &coordinates, Settings settings){
    SCAI_REGION( "SpectralPartition.getPartition" )
    	
    std::chrono::time_point<std::chrono::steady_clock> start, round;
    start = std::chrono::steady_clock::now();
    
    //const scai::dmemo::DistributionPtr coordDist = coordinates[0].getDistributionPtr();
    const scai::dmemo::DistributionPtr inputDist = adjM.getRowDistributionPtr();
    const scai::dmemo::CommunicatorPtr comm = inputDist->getCommunicatorPtr();
    
    IndexType k = settings.numBlocks;
    const IndexType dimensions = coordinates.size();
    const IndexType localN = inputDist->getLocalSize();
    const IndexType globalN = inputDist->getGlobalSize();

    if (k != comm->getSize() && comm->getRank() == 0) {
    	throw std::logic_error("Spectral partition only implemented for same number of blocks and processes.");
    }

    // get a pixeled-coarsen graph , this is replicated in every PE
    scai::lama::DenseVector<ValueType> pixelWeights;
    scai::lama::CSRSparseMatrix<ValueType> pixelGraph = MultiLevel<IndexType, ValueType>::pixeledCoarsen(adjM, coordinates, pixelWeights, settings);
    SCAI_ASSERT( pixelGraph.getRowDistributionPtr()->isReplicated() == 1, "Pixel graph should (?) be replicated.");
    
    IndexType numPixels = pixelGraph.getNumRows();
    SCAI_ASSERT( numPixels == pixelGraph.getNumColumns(), "Wrong pixeled graph.");
    SCAI_ASSERT( pixelGraph.isConsistent() == 1 , "Pixeled graph not consistent.");
    
    // get the laplacian of the pixeled graph , since the pixeled graph is replicated so should be the laplacian
    scai::lama::CSRSparseMatrix<ValueType> laplacian = GraphUtils::constructLaplacian<IndexType, ValueType>(pixelGraph);
    SCAI_ASSERT( laplacian.isConsistent() == 1 , "Laplacian graph not consistent.");
    SCAI_ASSERT( laplacian.getNumRows() == numPixels , "Wrong size of the laplacian.");
    
    if( !laplacian.getRowDistributionPtr()->isReplicated() ){
        // replicate the laplacian
        scai::dmemo::DistributionPtr pixelNoDistPointer(new scai::dmemo::NoDistribution( numPixels ));
        laplacian.redistribute( pixelNoDistPointer, pixelNoDistPointer);
    }else{
        PRINT0("Laplacian already replicated, no need to redistribute.");
    }

    // the fiedler vector corresponding to second smallest eigenvalue
    scai::lama::DenseVector<ValueType> fiedler;
    ValueType fiedlerEigenvalue;

    scai::lama::DenseVector<IndexType> permutation;
    
    {
        SCAI_REGION( "SpectralPartition.getPartition.getFiedlerVectorAndSort" )
        fiedler= SpectralPartition<IndexType, ValueType>::getFiedlerVector( pixelGraph, fiedlerEigenvalue );
        SCAI_ASSERT( fiedler.size() == numPixels, "Sizes do not agree.");
        fiedler.sort(permutation, true);
    }
    
    //TODO(?): a distributed version
    // since pixelGraph is replicated so is the permutation
    SCAI_ASSERT( true, permutation.getDistributionPtr()->isReplicated() );
    
    // TODO: change to localPixelPartition( numPixels, k-1); so that the last part get the remaining pixels (if any)
    // get local partition of the pixeled graph
    DenseVector<IndexType> localPixelPartition( numPixels, -1);
    IndexType averageBlockSize = globalN/k +1;
    IndexType serialPixelInd =0;
    
    for(IndexType block=0; block<k; block++){
        IndexType thisBlockSize = 0;
        // these two vectors should be replicated
        SCAI_ASSERT( localPixelPartition.getDistributionPtr()->isReplicated() == 1, "Should be (?) replicated.");
        SCAI_ASSERT( pixelWeights.getDistributionPtr()->isReplicated() == 1, "Should be (?) replicated.");
        
        scai::hmemo::WriteOnlyAccess<IndexType> wPixelPart( localPixelPartition.getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> rPixelWeights( pixelWeights.getLocalValues() );
        
        while( thisBlockSize < averageBlockSize){ 
            SCAI_ASSERT( serialPixelInd<permutation.getLocalValues().size(), "Pixel index " << serialPixelInd << " too big.");
            IndexType pixel= permutation.getLocalValues()[ serialPixelInd ];
            SCAI_ASSERT( pixel<numPixels, "Wrong pixel value "<< pixel);
            wPixelPart[ pixel ] = block;
            thisBlockSize += rPixelWeights[ pixel ];
            ++serialPixelInd;
            
            if(serialPixelInd >= numPixels){
                if(block != k-1){
                    PRINT("Pixels finished but still have blocks that will be empty: current block is "<< block << " and k= "<< k << std::endl << "This should not happen. Exiting...");
                    return DenseVector<IndexType>(inputDist, -1);
                }
                break;
            }
        }
        // every block can have size > averageBlockSize but hopefully not much more...
    }
    
    {
        // TODO: for debugging, should remove
        for(int i=0; i<numPixels; i++){
            scai::hmemo::ReadAccess<IndexType> rPixelPart( localPixelPartition.getLocalValues() );
            int b = rPixelPart[i];
            SCAI_ASSERT( b>=0, "Wrong pixel partitioning " << b <<" for pixel "<< i);
            SCAI_ASSERT( b<k, "Wrong pixel partitioning " << b <<" for pixel "<< i);
        }
    }
    //
    // here, every pixel must belong to a part
    //
    
    const unsigned long sideLen = settings.pixeledSideLen;
    
    //TODO: finding the max is also done in pixeledCoarsen, maybe we can pass max as input parameter there
    std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
    
    // get local max
    for (IndexType dim = 0; dim < dimensions; dim++) {
        //get local parts of coordinates
        const scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[dim].getLocalValues();
        for (IndexType i = 0; i < localN; i++) {
            ValueType coord = localPartOfCoords[i];
            if (coord > maxCoords[dim]) maxCoords[dim] = coord;
        }
    }
    // communicate to get global  max
    for (IndexType dim = 0; dim < dimensions; dim++) {
        maxCoords[dim] = comm->max(maxCoords[dim]);
    }
    
    // set your local part of the partition/result
    DenseVector<IndexType>  result(inputDist, 0);
    
    {   
        scai::hmemo::WriteOnlyAccess<IndexType> wLocalPart ( result.getLocalValues() );
        scai::hmemo::ReadAccess<IndexType> rPixelPart( localPixelPartition.getLocalValues() );
        
        scai::hmemo::ReadAccess<ValueType> coordAccess0( coordinates[0].getLocalValues() );
        scai::hmemo::ReadAccess<ValueType> coordAccess1( coordinates[1].getLocalValues() );
        // this is faulty, if dimensions=2 coordAccess2 is equal to coordAccess1
        scai::hmemo::ReadAccess<ValueType> coordAccess2( coordinates[dimensions-1].getLocalValues() );
        
        for(IndexType i=0; i<localN; i++){
            IndexType scaledX = sideLen*coordAccess0[i]/(maxCoords[0]+1);
            IndexType scaledY = sideLen*coordAccess1[i]/(maxCoords[1]+1);
            IndexType thisPixel;    // the pixel this node belongs to
            if(dimensions==3){
                IndexType scaledZ = sideLen*coordAccess2[i]/(maxCoords[2]+1);
                thisPixel = scaledX*sideLen*sideLen + scaledY*sideLen + scaledZ;
            } else{
                thisPixel = scaledX*sideLen + scaledY;
            }
            SCAI_ASSERT( thisPixel < numPixels, "Index too big: "<< thisPixel );
            
            // set the block for this node
            wLocalPart[i] = rPixelPart[ thisPixel ];
        }
    }

    return result;
}
//---------------------------------------------------------------------------------------


template<typename IndexType, typename ValueType>
scai::lama::DenseVector<ValueType> SpectralPartition<IndexType, ValueType>::getFiedlerVector(const scai::lama::CSRSparseMatrix<ValueType>& adjM, ValueType& eigenvalue ){
    SCAI_REGION("SpectralPartition.getFiedlerVector");
    
    IndexType globalN= adjM.getNumRows();
    SCAI_ASSERT_EQ_ERROR( globalN, adjM.getNumColumns(), "Matrix not square, numRows != numColumns");
    
    scai::lama::CSRSparseMatrix<ValueType> laplacian = GraphUtils::constructLaplacian<IndexType, ValueType>( adjM );
    
    // set u=[ 1+sqrt(n), 1, 1, 1, ... ]
    ValueType n12 = scai::common::Math::sqrt( ValueType( globalN ));
    
    scai::lama::DenseVector<ValueType> u( laplacian.getRowDistributionPtr(), 1);
    u[0] = n12 + 1;
    
    scai::lama::Scalar alpha= globalN + n12;
    
    // set h= L*u/alpha
    scai::lama::DenseVector<ValueType> h( laplacian*u );
    h /= alpha;
    
    // set v= h - gamma*u/2
    scai::lama::Scalar gamma= u.dotProduct(h)/ alpha * 0.5;
    scai::lama::DenseVector<ValueType> v(h - gamma*u);
    
    scai::lama::DenseVector<ValueType> r(u); r[0]= 0.0;
    scai::lama::DenseVector<ValueType> s(v); s[0]= 0.0;
    scai::lama::DenseVector<ValueType> t( laplacian.getRowDistributionPtr(), 1.0);
    
    scai::lama::DenseVector<ValueType> y;
    scai::lama::DenseVector<ValueType> diff;
    
    t[0] = 0.0;
    scai::lama::DenseVector<ValueType> z(t);
    
    IndexType kmax = 220;        // maximal number of iterations
    Scalar    eps  = 1e-7;       // accuracy for maxNorm
    scai::lama::Scalar lambda = 0.0;   // the eigenvalue (?) TODO: make sure
    
    for(IndexType k=0; k<kmax; k++){
        // normalize t
        t = t/t.l2Norm();
        
        y= laplacian*t;
        y[0] = 0.0;                  // fill element as we actually use L[2:n,2:n] 
        y -= s.dotProduct( t ) * r;  
        y -= r.dotProduct( t ) * s;
        
        lambda = t.dotProduct( y );
        diff = y - lambda * t;
        Scalar diffNorm = diff.maxNorm();
        
        if( diffNorm<eps){
            break;
        }
        
        //solve
        //set res
        scai::lama::DenseVector<ValueType> res ( t- laplacian*z );
        res += s.dotProduct(z) * r;
        res += r.dotProduct(z) * s;
        res[0] = 0.0;
        
        scai::lama::DenseVector<ValueType> d(res);
        
        scai::lama::Scalar rOld = res.dotProduct( res );
        scai::lama::L2Norm norm;
        scai::lama::Scalar rNorm = norm(res);
        
        IndexType maxIter= 100;
        
        for(IndexType kk=0; kk<maxIter and rNorm>eps; kk++){
            scai::lama::DenseVector<ValueType> x( laplacian*d );
            
            x -= s.dotProduct(d) * r;
            x -= r.dotProduct(d) * s;
            x[0] = 0.0;
            
            scai::lama::Scalar tmpSc = rOld/ d.dotProduct(x);
            z = z + tmpSc*d;
            res = res - tmpSc*x;
            scai::lama::Scalar rNew = res.dotProduct(res);
            scai::lama::Scalar beta = rNew/ rOld;
            d = res + beta*d;
            rOld = rNew;
            rNorm = norm( res );
        }
      
        t = z;
    }
 
    t[0] = 0.0;
    scai::lama::Scalar beta = u.dotProduct(t) / alpha;
    t = t- beta*u;
    
    eigenvalue = lambda.Scalar::getValue<ValueType>();
    
    return t;
}

//---------------------------------------------------------------------------------------

template class SpectralPartition<IndexType, ValueType>;

};
