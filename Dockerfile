FROM ubuntu:focal

WORKDIR /signon_slr

# Install the required applications and system libraries using apt.
# Avoid that tzdata installation asks for the time zone.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3.9 python3-pip python3-opencv

# Set up a non-root user for inside the container.
# -m: creates home directory.
# -s: uses /bin/bash as login shell.
RUN useradd -ms /bin/bash signon
USER signon
WORKDIR /home/signon
# Add the directory where pip will install libraries to path.
ENV PATH="/home/signon/.local/bin:${PATH}"

# Install the required Python dependencies using pip.
ADD requirements.txt .
RUN pip install --user --upgrade pip
RUN pip install --user -r requirements.txt
ADD web_service/feature_extractor/slr_pipeline/requirements.txt .
RUN pip install --user -r requirements.txt

# Set some environment variables that can be used by scripts running inside the container.
ENV SIGNON_SLR_COMPONENT_VERSION=0.5.0

# Copy over the required files for the web service.
COPY web_service web_service
WORKDIR /home/signon/web_service
# Make sure the signon user can execute the launch script.
USER root
RUN chmod +x launch.sh
USER signon

# Launch the web service.
ENTRYPOINT ./launch.sh
