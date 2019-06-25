# Use an official Ubuntu runtime as a parent image
FROM ubuntu:16.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages
RUN apt update && apt install -y \
	cmake \
	g++ \
	git \
	libatlas-base-dev \
	mpi-default-dev

# Clone Lama
RUN git clone https://github.com/kit-parco/lama.git

# Configure Lama
RUN cd lama && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../install ../scai && cd ../..

# Compile Lama
RUN cd lama/build && make -j8 && make install && cd ../..

# Configure Geographer
RUN mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=../install \
	-DSCAI_DIR=/app/lama/install/ .. && cd ..

# Compile Geographer
RUN cd build && make -j8 && make install && cd ..
