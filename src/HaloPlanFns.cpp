
#include "HaloPlanFns.h"

#include <scai/tracing.hpp>

#include <set>

using namespace scai;
using namespace dmemo;
using namespace hmemo;

namespace ITI {

scai::dmemo::HaloExchangePlan coarsenHalo(
    const Distribution& coarseDistribution,
    const HaloExchangePlan& halo,
    const HArray<IndexType>& localFineToCoarse,
    const HArray<IndexType>& haloFineToCoarse )
{
    SCAI_REGION( "HaloExchangePlan.coarsenHalo" )

    ReadAccess<IndexType> providedIndices(halo.getLocalIndexes());
    ReadAccess<IndexType> requiredIndices(halo.getHalo2GlobalIndexes());
    scai::dmemo::CommunicationPlan sendPlan = halo.getLocalCommunicationPlan();
    scai::dmemo::CommunicationPlan recvPlan = halo.getHaloCommunicationPlan();

    SCAI_ASSERT(providedIndices.size() == sendPlan.totalQuantity(), "Communication plan does not fit provided indices.");
    SCAI_ASSERT(requiredIndices.size() == recvPlan.totalQuantity(), "Communication plan does not fit required indices.");

    std::vector<IndexType> newProvidedIndices;
    std::vector<IndexType> sendQuantities;

    {
        ReadAccess<IndexType> rFineToCoarse(localFineToCoarse);

        //construct new send plan
        for (IndexType i = 0; i < sendPlan.size(); i++)
        {
            scai::dmemo::CommunicationPlan::Entry entry = sendPlan[i];

            if (IndexType(sendQuantities.size()) <= entry.partitionId)
            {
                sendQuantities.resize(entry.partitionId+1);
            }

            std::set<IndexType> sendSet;

            for (IndexType j = entry.offset; j < entry.offset + entry.quantity; j++)
            {
                SCAI_ASSERT(j < providedIndices.size(), "Communication plan does not fit provided indices.");
                IndexType provIndex = providedIndices[j];
                SCAI_ASSERT(provIndex < rFineToCoarse.size(), "Provided index " << provIndex << " seemingly not local.");
                sendSet.insert(coarseDistribution.global2Local(rFineToCoarse[providedIndices[j]]));
            }

            newProvidedIndices.insert(newProvidedIndices.end(), sendSet.begin(), sendSet.end());
            sendQuantities[entry.partitionId] = sendSet.size();
        }
    }
    SCAI_ASSERT(IndexType(newProvidedIndices.size()) <= providedIndices.size(), "New index list is bigger than old one.");

    auto coarseSendPlan = CommunicationPlan( sendQuantities );
    // coarseHalo.mProvidesPlan.allocate(sendQuantities.data(), sendQuantities.size());

    SCAI_ASSERT_EQ_ERROR( coarseSendPlan.totalQuantity(), IndexType(newProvidedIndices.size()), "serious mismatch" )

    std::vector<IndexType> newRequiredIndices;
    std::vector<IndexType> recvQuantities;

    {
        ReadAccess<IndexType> rFineToCoarse(haloFineToCoarse);

        //construct new recv plan
        for (IndexType i = 0; i < recvPlan.size(); i++)
        {
            scai::dmemo::CommunicationPlan::Entry entry = recvPlan[i];

            if (IndexType(recvQuantities.size()) <= entry.partitionId)
            {
                recvQuantities.resize(entry.partitionId+1);
            }

            std::set<IndexType> recvSet;

            for (IndexType j = entry.offset; j < entry.offset + entry.quantity; j++)
            {
                IndexType reqIndex = requiredIndices[j];
                SCAI_ASSERT(halo.global2Halo(reqIndex) != invalidIndex, "Index" << reqIndex << " seemingly not in halo");
                SCAI_ASSERT(halo.global2Halo(reqIndex) < rFineToCoarse.size(), "Index" << halo.global2Halo(reqIndex) << " too big for halo data");
                recvSet.insert(rFineToCoarse[halo.global2Halo(requiredIndices[j])]);
            }

            for (IndexType reqIndex : recvSet)
            {
                newRequiredIndices.push_back(reqIndex);
            }

            recvQuantities[entry.partitionId] = recvSet.size();
        }
    }

    SCAI_ASSERT_LE_ERROR( IndexType(newRequiredIndices.size()), requiredIndices.size(),
                          "New index list is bigger than old one.");

    auto coarseRecvPlan = CommunicationPlan( recvQuantities );

    SCAI_ASSERT_EQ_ERROR( coarseRecvPlan.totalQuantity(), IndexType( newRequiredIndices.size() ), "serious mismatch" )

    return HaloExchangePlan( HArray<IndexType>( newRequiredIndices ),
                             HArray<IndexType>( newProvidedIndices ),
                             std::move( coarseRecvPlan ),
                             std::move( coarseSendPlan ) );
}

/* ---------------------------------------------------------------------- */

HaloExchangePlan buildWithPartner(
    const Distribution& distribution,
    const HArray<IndexType>& requiredIndexes,
    const HArray<IndexType>& providedIndexes,
    const PartitionId partner )
{
    SCAI_REGION( "HaloBuilder.buildWithPartner" )

    auto requiredPlan = CommunicationPlan::buildSingle( requiredIndexes.size(), partner );
    auto providesPlan = CommunicationPlan::buildSingle( providedIndexes.size(), partner );

    HArray<IndexType> localIndexes;

    distribution.global2LocalV( localIndexes, providedIndexes );

    return HaloExchangePlan( requiredIndexes,
                             std::move( localIndexes ),
                             std::move( requiredPlan ),
                             std::move( providesPlan ) );
}

/* ---------------------------------------------------------------------- */

}
