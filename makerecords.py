import socket, sys, thread
from struct import *

#create a STREAMing socket
def capture():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
    except socket.error , msg:
        print 'Socket could not be created. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
        sys.exit()

    # receive a packet
    f = open('test.csv', 'wb')
    while True:
        packet = s.recvfrom(65565)

        #packet string from tuple
        packet = packet[0]

        #take first 20 characters for the ip header
        ip_header = packet[0:20]

        #now unpack the packets
        iph = unpack('!BBHHHBBH4s4s' , ip_header)

        version_ihl = iph[0]
        version = version_ihl >> 4
        ihl = version_ihl & 0xF

        iph_length = ihl * 4

        ttl = iph[5]
        protocol = iph[6]
        if int(protocol) == 6 :
            protocol = "tcp"
        s_addr = socket.inet_ntoa(iph[8]);
        d_addr = socket.inet_ntoa(iph[9]);

        #print str(protocol) + ',' + str(s_addr) + ',' + str(d_addr)

        tcp_header = packet[iph_length:iph_length+20]

        #now unpack them :)
        tcph = unpack('!HHLLBBHHH' , tcp_header)

        source_port = tcph[0]
        dest_port = tcph[1]
        if int(dest_port) == 22:
            dest_port = "ssh"
        sequence = tcph[2]
        acknowledgement = tcph[3]
        doff_reserved = tcph[4]
        tcph_length = doff_reserved >> 4

        f.write(str(protocol) + ',' + str(s_addr) + ',' + str(d_addr)+',' + str(source_port) + ',' + str(dest_port) + ',' + str(sequence) + ',' + str(acknowledgement) + ',' + str(tcph_length)+ '\n')
