#pragma once

#include <scai/dmemo/HaloExchangePlan.hpp>

namespace ITI {

scai::dmemo::HaloExchangePlan coarsenHalo(
    const scai::dmemo::Distribution& coarseDistribution,
    const scai::dmemo::HaloExchangePlan& halo,
    const scai::hmemo::HArray<scai::IndexType>& localFineToCoarse,
    const scai::hmemo::HArray<scai::IndexType>& haloFineToCoarse );

scai::dmemo::HaloExchangePlan buildWithPartner(
    const scai::dmemo::Distribution& distribution,
    const scai::hmemo::HArray<scai::IndexType>& requiredIndexes,
    const scai::hmemo::HArray<scai::IndexType>& providedIndexes,
    const scai::PartitionId partner );

}

