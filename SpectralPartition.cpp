/*
 * SpectralPartition
 *
 *  Created on: 15.03.17
 *      Author: tzovas
 */


#include "SpectralPartition.h"

namespace ITI {


template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> SpectralPartition<IndexType, ValueType>::getDegreeVector( const scai::lama::CSRSparseMatrix<ValueType>& adjM){
    SCAI_REGION("SpectralPartition.getDegreeVector");
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    const IndexType localN = distPtr->getLocalSize();
    
    scai::lama::DenseVector<IndexType> degreeVector(distPtr);
    scai::utilskernel::LArray<IndexType>& localDegreeVector = degreeVector.getLocalValues();
    
    const scai::lama::CSRStorage<ValueType> localAdjM = adjM.getLocalStorage();
    {
        const scai::hmemo::ReadAccess<IndexType> readIA ( localAdjM.getIA() );
        scai::hmemo::WriteOnlyAccess<IndexType> writeVector( localDegreeVector, localDegreeVector.size()) ;
        
        SCAI_ASSERT_EQ_ERROR(readIA.size(), localDegreeVector.size()+1, "Probably wrong distribution");
        
        for(IndexType i=0; i<readIA.size()-1; i++){
            writeVector[i] = readIA[i+1] - readIA[i];
        }
    }
    
    return degreeVector;
}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::CSRSparseMatrix<ValueType> SpectralPartition<IndexType, ValueType>::getLaplacian( const scai::lama::CSRSparseMatrix<ValueType>& adjM){
    SCAI_REGION("SpectralPartition.getLaplacian");
    
    scai::dmemo::CommunicatorPtr comm = scai::dmemo::Communicator::getCommunicatorPtr();
    const scai::dmemo::DistributionPtr distPtr = adjM.getRowDistributionPtr();
    
    const IndexType globalN = distPtr->getGlobalSize();
    const IndexType localN = distPtr->getLocalSize();
    
    const CSRStorage<ValueType>& localStorage = adjM.getLocalStorage();
    
    // vector of size globalN with the degree for every edge
    scai::lama::DenseVector<IndexType> degreeVector = SpectralPartition<IndexType, ValueType>::getDegreeVector( adjM );
    SCAI_ASSERT( degreeVector.size() == globalN, "Degree vector global size not correct: " << degreeVector.size() << " , shoulb be " << globalN);
    SCAI_ASSERT( degreeVector.getLocalValues().size() == localN,"Degree vector local size not correct: " << degreeVector.getLocalValues().size() << " , shoulb be " << localN);
    
    // data of the output graph
    scai::hmemo::HArray<IndexType> laplacianIA;
    scai::hmemo::HArray<IndexType> laplacianJA;
    scai::hmemo::HArray<ValueType> laplacianValues;
    
    IndexType laplacianNnzValues;
    {        
        // get local data of adjM
        scai::hmemo::ReadAccess<IndexType> ia(localStorage.getIA());
        scai::hmemo::ReadAccess<IndexType> ja(localStorage.getJA());
        scai::hmemo::ReadAccess<ValueType> values(localStorage.getValues());
        
        // local data of dree vector
        scai::hmemo::ReadAccess<IndexType>  rLocalDegree( degreeVector.getLocalValues() );
        assert( degreeVector.getLocalValues().size() == localN );

        laplacianNnzValues = values.size() + localN;    // add one element per node/row
        
        // data of laplacian graph. laplacian and input are of the same size globalN x globalN
        scai::hmemo::WriteOnlyAccess<IndexType> wLaplacianIA( laplacianIA , ia.size() );  
        scai::hmemo::WriteOnlyAccess<IndexType> wLaplacianJA( laplacianJA , laplacianNnzValues );
        scai::hmemo::WriteOnlyAccess<ValueType> wLaplacianValues( laplacianValues, laplacianNnzValues );
        
   /*     
        IndexType localInd = 0;
        IndexType globalIndex = distPtr->local2Global( localInd );
        //IndexType laplacianOffset = 0;
        IndexType laplacianIndex = 0;
        
        for(IndexType j=0; j<ja.size(); j++){
            SCAI_ASSERT( ja[j]==globalIndex, "Diagonal no empty, no self loops allowed.");
            laplacianIndex = j + localInd;
            if( ja[j]<globalIndex ){
                wLaplacianJA[laplacianIndex] = ja[j];
                wLaplacianValues[laplacianIndex] = -1*values[j];    //opposite value
            }else{      //add diagonal element
                wLaplacianJA[laplacianIndex] = globalIndex;
                assert( localInd < localocalDegree.size() );
                wLaplacianValues[laplacianIndex] = rLocalDegree[ localInd ];
                ++localInd;
                // can we write ++globalindex; ? Probably
                if( localInd >localN){      //for the last local element
                    globalIndex = globalN;
                }else{
                    globalIndex = distPtr->local2Global( localInd );
                }
            }
            if( j+1<ja.size() and j+1> ia[localInd+1] ){  //changed row
                globalIndex = distPtr->local2Global( localInd );    // or just ++ ?
            }
            
        }
   */     

        
        IndexType nnzCounter = 0;
        for(IndexType i=0; i<localN; i++){
            const IndexType beginCols = ia[i];
            const IndexType endCols = ia[i+1];
            assert(ja.size() >= endCols);
            
            //IndexType neighbor = ja[j];  // neighbor of node i (global indexing)
            IndexType globalI = distPtr->local2global(i);
            IndexType j = beginCols;
            
            while( ja[j]< globalI and j<endCols){     //bot-left part of matrix, before diagonal
                assert(ja[j] >= 0);
                assert(ja[j] < globalN);
                
                wLaplacianJA[nnzCounter] = ja[j];          // same indices
                wLaplacianValues[nnzCounter] = -values[j]; // opposite values
                ++nnzCounter;
                assert( nnzCounter < laplacianNnzValues+1);
                ++j;
            }
            // out of while, must insert diagonal element
            wLaplacianJA[nnzCounter] = globalI;
            assert( i < rLocalDegree.size() );
            wLaplacianValues[nnzCounter] = rLocalDegree[i];
//PRINT(*comm << ": "<< i << " _ "<< nnzCounter << " >> "<< wLaplacianJA[nnzCounter] << ", "<< wLaplacianValues[nnzCounter] );        
            ++nnzCounter;
            // copy the rest of the row
            while( j<endCols){
                wLaplacianJA[nnzCounter] = ja[j];          // same indices
                wLaplacianValues[nnzCounter] = -values[j]; // opposite values
                ++nnzCounter;
                assert( nnzCounter < laplacianNnzValues+1);
                ++j;
            }
            
        }
        
        //fix ia array , we add 1 element in every row
        for(IndexType i=0; i<ia.size(); i++){
            wLaplacianIA[i] = ia[i] + i;
        }

    }
    
    SCAI_ASSERT_EQ_ERROR(laplacianJA.size(), laplacianValues.size(), "Wrong sizes." );
    {
        scai::hmemo::ReadAccess<IndexType> rLaplacianIA( laplacianIA );
        scai::hmemo::ReadAccess<IndexType> rLaplacianJA( laplacianJA );
        scai::hmemo::ReadAccess<ValueType> rLaplacianValues( laplacianValues );
        
        SCAI_ASSERT_EQ_ERROR(rLaplacianIA[ rLaplacianIA.size()-1] , laplacianJA.size(), "Wrong sizes." );
        /*
        for(int i=0; i<laplacianJA.size(); i++){
            PRINT(*comm <<" = " <<rLaplacianValues[i]);
        }
        */
    }
    
    scai::lama::CSRStorage<ValueType> resultStorage( localN, globalN, laplacianNnzValues, laplacianIA, laplacianJA, laplacianValues);
    
    scai::lama::CSRSparseMatrix<ValueType> result(adjM.getRowDistributionPtr() , adjM.getColDistributionPtr() );
    result.swapLocalStorage( resultStorage );
    
    return result;

}
//---------------------------------------------------------------------------------------

template<typename IndexType, typename ValueType>
scai::lama::DenseVector<IndexType> SpectralPartition<IndexType, ValueType>::getPartition(CSRSparseMatrix<ValueType> &adjM, std::vector<DenseVector<ValueType>> &coordinates, Settings settings){
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

    // get a pixeled-coarsen graph , this is replicated in every PE
    scai::lama::DenseVector<IndexType> pixelWeights;
    scai::lama::CSRSparseMatrix<ValueType> pixelGraph = MultiLevel<IndexType, ValueType>::pixeledCoarsen(adjM, coordinates, pixelWeights, settings);
    SCAI_ASSERT( pixelGraph.getRowDistributionPtr()->isReplicated() == 1, "Pixel graph should (?) be replicated.");
    
    IndexType numPixels = pixelGraph.getNumRows();
    SCAI_ASSERT( numPixels == pixelGraph.getNumColumns(), "Wrong pixeled graph.");
    SCAI_ASSERT( pixelGraph.isConsistent() == 1 , "Pixeled graph not consistent.");
    
    // get the laplacian of the pixeled graph , since the pixeled graph is replicated so should be the laplacian
    scai::lama::CSRSparseMatrix<ValueType> laplacian = SpectralPartition<IndexType, ValueType>::getLaplacian( pixelGraph );
    SCAI_ASSERT( laplacian.isConsistent() == 1 , "Laplacian graph not consistent.");
    SCAI_ASSERT( laplacian.getNumRows() == numPixels , "Wrong size of the laplacian.");
    
    if( !laplacian.getRowDistributionPtr()->isReplicated() ){
        // replicate the laplacian
        scai::dmemo::DistributionPtr pixelNoDistPointer(new scai::dmemo::NoDistribution( numPixels ));
        laplacian.redistribute( pixelNoDistPointer, pixelNoDistPointer);
    }else{
        PRINT0("Laplacian already replicated, no need to redistribute.");
    }
    
    //
    // From down here a big part (until ^^^) is local/replicated in every PE
    //
    
    // get the second eigenvector of the laplacian (local, not distributed)
    //TODO: if local, change to std::vector
    DenseVector<ValueType> eigenVec (numPixels, -1);
    {
        using Eigen::MatrixXd;
        using namespace Eigen;
        
        // copy to an eigen::matrix
        MatrixXd eigenLapl( numPixels, numPixels);
        assert(numPixels == laplacian.getNumRows());
        for( int r=0; r<laplacian.getNumRows(); r++){
            for( int c=0; c<laplacian.getNumColumns(); c++){
                eigenLapl(c,r) = laplacian.getValue( r, c).Scalar::getValue<ValueType>();
            }
        }
    
        // solve and get the second eigenvector
        SelfAdjointEigenSolver<MatrixXd> eigensolver( eigenLapl );
        VectorXd secondEigenVector = eigensolver.eigenvectors().col(2) ;
        SCAI_ASSERT( secondEigenVector.size() == numPixels, "Sizes do not agree.");
        
        // copy to DenseVector
        for(int i=0; i<secondEigenVector.size(); i++){
            eigenVec.setValue( i, secondEigenVector[i]);
        }
    }
    //redistribute the eigenVec
    //eigenVec.redistribute( inputDist );
    
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ redistributing the eigen vector
    
    // we could sort locally, no need to redistribute
    // doing it like this to mimic real case senario
    
    // sort
    //TODO: if local, change to std::vector
    scai::lama::DenseVector<IndexType> permutation;
    eigenVec.sort(permutation, true);
    
    // TODO: change to localPixelPartition( numPixels, k-1); so last part get the remaining pixels (if any)
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
        scai::hmemo::ReadAccess<IndexType> rPixelWeights( pixelWeights.getLocalValues() );
        
        while( thisBlockSize < averageBlockSize){ 
            SCAI_ASSERT( serialPixelInd<permutation.getLocalValues().size(), "Pixel index too big.");
            IndexType pixel= permutation.getLocalValues()[ serialPixelInd ];
            SCAI_ASSERT( pixel<numPixels, "Wrong pixel value "<< pixel);
            wPixelPart[ pixel ] = block;
            thisBlockSize += rPixelWeights[ pixel ];
            ++serialPixelInd;
            
            if(serialPixelInd >= numPixels){
                if(block != k-1){
                    PRINT("Pixels finished but still have blocks that will be empty." << std::endl << "This should not happen. Exiting...");
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
    
    const unsigned int detailLvl = settings.pixeledDetailLevel;
    const unsigned long sideLen = std::pow(2,detailLvl);
    
    //TODO: finding the max is also done in pixeledCoarsen, maybe we can pass max as input parameter there
    std::vector<ValueType> maxCoords(dimensions, std::numeric_limits<ValueType>::lowest());
    
    // get local max
    for (IndexType dim = 0; dim < dimensions; dim++) {
        //get local parts of coordinates
        scai::utilskernel::LArray<ValueType>& localPartOfCoords = coordinates[dim].getLocalValues();
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
    /*
    DenseVector<IndexType>  result;
    
    if (!inputDist->isReplicated() && comm->getSize() == k) {
        SCAI_REGION( "SpectralPartition.getPartition.redistribute" )
        
        //scai::dmemo::DistributionPtr blockDist(new scai::dmemo::BlockDistribution(globalN, comm));
        //permutation.redistribute(blockDist);
        //scai::hmemo::WriteAccess<IndexType> wPermutation( permutation.getLocalValues() );
        //std::sort(wPermutation.get(), wPermutation.get()+wPermutation.size());
        //wPermutation.release();
        
        scai::dmemo::DistributionPtr newDistribution(new scai::dmemo::GeneralDistribution(globalN, permutation.getLocalValues(), comm));
        
        adjM.redistribute( newDistribution, adjM.getColDistributionPtr());
        result = DenseVector<IndexType>(newDistribution, comm->getRank());
        
        if (settings.useGeometricTieBreaking) {
            for (IndexType dim = 0; dim < dimensions; dim++) {
                coordinates[dim].redistribute(newDistribution);
            }
        }
        
    }
    */
    
    //
    // redistribute based on the new partition
    
    //TODO: to create the new distribution must replicate the result (aka partition) vector. This is not good.
    //      Maybe gather in one root PE and scatter later or find another way...
    scai::dmemo::DistributionPtr noDist (new scai::dmemo::NoDistribution( globalN ));
    result.redistribute(noDist);
    assert( result.getDistribution().isReplicated() );    
    
    scai::dmemo::DistributionPtr newDist(new scai::dmemo::GeneralDistribution( result.getLocalValues(), comm));
    result.redistribute( newDist);

    adjM.redistribute(newDist, adjM.getColDistributionPtr());
    
    // redistibute coordinates
    for (IndexType dim = 0; dim < dimensions; dim++) {
          coordinates[dim].redistribute( newDist );
    }    
    // check coordinates size
    for (IndexType dim = 0; dim < dimensions; dim++) {
        assert( coordinates[dim].size() == globalN);
        assert( coordinates[dim].getLocalValues().size() == newDist->getLocalSize() );
    }

    ValueType cut = comm->getSize() == 1 ? ParcoRepart<IndexType, ValueType>::computeCut(adjM, result) : comm->sum(ParcoRepart<IndexType, ValueType>::localSumOutgoingEdges(adjM, false)) / 2;
    ValueType imbalance = ParcoRepart<IndexType, ValueType>::computeImbalance(result, k);
    if (comm->getRank() == 0) {
        IndexType detailLvl = settings.pixeledDetailLevel;
        std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() -start;
        std::cout << "\033[1;32mSpectral partition"<<" (" << elapsedSeconds.count() << " seconds), cut is " << cut << std::endl;
        std::cout<< "and imbalance= "<< imbalance << "\033[0m"  << std::endl;
    }
    
    return result;
}
//---------------------------------------------------------------------------------------


template scai::lama::DenseVector<int> SpectralPartition<int, double>::getDegreeVector( const scai::lama::CSRSparseMatrix<double>& adjM);

template scai::lama::CSRSparseMatrix<double> SpectralPartition<int, double>::getLaplacian( const scai::lama::CSRSparseMatrix<double>& adjM);

template scai::lama::DenseVector<int> SpectralPartition<int, double>::getPartition(CSRSparseMatrix<double> &adjM, std::vector<DenseVector<double>> &coordinates, Settings settings);


};