INCLUDES = -I/usr/local/include/opencv
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
LIBDIRS = -L/usr/local/lib

OPT = -O3 -Wno-deprecated

CC=g++

.PHONY: all clean

OBJS= main.o

all:main
	echo all:make complete

clean:
	rm -f *.o *~ main

%.o:%.cpp
	$(CC) -c $(INCLUDES) $+ $(OPT)

main:$(OBJS)
	$(CC) $(LIBDIRS) $(LIBS) -o $@ $+ $(OPT)
