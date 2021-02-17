#pragma once

//#include <mpi.h>
#include <string.h>

#include "sys/types.h"
#include "sys/sysinfo.h"
#include "sys/times.h"
#include "sys/vtimes.h"

#include "Settings.h"

//taken from 
//https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
//returns the ram this PE uses and the free ram of the compute node
inline
std::pair<double,double> getFreeRam(const scai::dmemo::CommunicatorPtr& comm, double& freeRam, bool printMessage=false){

    struct sysinfo memInfo;
    const double kb = 1024.0;
    const double mb = kb*1024;
    [[maybe_unused]] const double gb = mb*1024;

    sysinfo (&memInfo);
    long long totalVirtualMem = memInfo.totalram;
    //Add other values in next statement to avoid int overflow on right hand side...
    totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;

    long long totalPhysMem = memInfo.totalram;
    //Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;

    long long physMemUsed = memInfo.totalram - memInfo.freeram;
    //Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;

    freeRam = memInfo.freeram;
    freeRam *= memInfo.mem_unit;

    unsigned long long sharedRam = memInfo.sharedram;
    sharedRam *= memInfo.mem_unit;    
    
    unsigned long long buffRam = memInfo.bufferram;
    buffRam *= memInfo.mem_unit; 

    auto parseLine = [](char* line){
        // This assumes that a digit will be found and the line ends in " Kb".
        int i = strlen(line);
        const char* p = line;
        while (*p <'0' || *p > '9') p++;
        line[i-3] = '\0';
        i = atoi(p);
        return i;
    };

    //Note: this value is in KB!
    auto getValue = [&](){ 
        FILE* file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != NULL){
            if (strncmp(line, "VmRSS:", 6) == 0){
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
        return result;
    };

    const double myMemUse = getValue()/kb;
    double totalMemUse = physMemUsed/mb;

    if( printMessage ){
        int rank = comm->getRank();
        double maxMem = comm->max(myMemUse);
        if( rank==0 ){
            std::cout<< "totalPhysMem: " << (totalPhysMem/mb) << 
                    " MB, physMemUsed: " << totalMemUse << 
                    " MB, free ram: " << freeRam/mb << 
                    " max mem used: " << maxMem << " MB" << std::endl;
        }
    }

    return std::make_pair(myMemUse,totalMemUse);
}