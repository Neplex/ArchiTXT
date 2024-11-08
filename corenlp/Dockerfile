ARG JAVA_VERSION=8u272
FROM openjdk:${JAVA_VERSION}

# Install wget and unzip to handle CoreNLP and models download/unpacking
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

ARG CORENLP_VERSION=4.5.7
ARG CORENLP_URL="http://nlp.stanford.edu/software/"
ARG CORENLP_HOME=/opt/corenlp

# Download and install CoreNLP
RUN wget ${MODELS_URL}stanford-corenlp-${CORENLP_VERSION}.zip && \
    unzip stanford-corenlp-${CORENLP_VERSION}.zip && \
    mv stanford-corenlp-${CORENLP_VERSION} ${CORENLP_HOME} && \
    rm stanford-corenlp-${CORENLP_VERSION}.zip

# Add models
ADD ${MODELS_URL}stanford-corenlp-${CORENLP_VERSION}-models-english.jar ${CORENLP_HOME}
ADD ${MODELS_URL}stanford-corenlp-${CORENLP_VERSION}-models-french.jar ${CORENLP_HOME}

# Set up the CLASSPATH for CoreNLP (collect all jar files in the directory)
RUN cd ${CORENLP_HOME} && \
    export CLASSPATH=""; \
    for file in $(find . -name "*.jar"); do \
        export CLASSPATH="$CLASSPATH:$(realpath $file)"; \
    done

# Expose the CoreNLP server port
EXPOSE 9000

# Set the working directory to CoreNLP
WORKDIR ${CORENLP_HOME}

# Set the default command to start the CoreNLP server with French properties
CMD java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
    -serverProperties StanfordCoreNLP-french.properties \
    -port 9000 \
    -timeout 15000
