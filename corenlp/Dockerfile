FROM openjdk:8u272

ENV VERSION=4.2.0

RUN	wget http://nlp.stanford.edu/software/stanford-corenlp-${VERSION}.zip; \
    unzip stanford-corenlp-${VERSION}.zip; \
    mv stanford-corenlp-${VERSION} /opt/corenlp

RUN wget http://nlp.stanford.edu/software/stanford-corenlp-${VERSION}-models-french.jar; \
	mv stanford-corenlp-${VERSION}-models-french.jar /opt/corenlp

RUN cd CoreNLP; \
	export CLASSPATH=""; for file in `find . -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done

EXPOSE 9000

WORKDIR /opt/corenlp

CMD java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-french.properties -port 9000 -timeout 15000