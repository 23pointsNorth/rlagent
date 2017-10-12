#!/usr/bin/python

import thread
import time
import socket
import struct
import sys
import future
import yaml

mcast_multicast_group = '239.1.1.1'
mcast_server_address = ('', 10000)
udp_server_address = ('127.0.0.1', 14555)

# Grrr


def send_message(udp_server_address, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:

        # Send data
        print >>sys.stderr, 'sending "%s"' % message
        sent = sock.sendto(message, udp_server_address)

        # Receive response
        print >>sys.stderr, 'waiting to receive'
        data, server = sock.recvfrom(4096)
        print >>sys.stderr, 'received "%s"' % data

    finally:
        print >>sys.stderr, 'closing socket'
        sock.close()


def mcast_listening_thread(multicast_group, server_address):
    # Create the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind to the server address
    sock.bind(server_address)

    # on all interfaces.
    group = socket.inet_aton(multicast_group)
    mreq = struct.pack('4sL', group, socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # Receive/respond loop
    while True:
        #print >>sys.stderr, '\nwaiting to receive message'
        data, address = sock.recvfrom(1024)

        #print >>sys.stderr, 'received %s bytes from %s' % (len(data), address)
        #print >>sys.stderr, data

        # Process specific messages
        name = str(data).split(None, 1)[0]
        core = str(data).split(None, 1)[1]
        msg = yaml.load(core)

        # Look for specific messages
        if name == 'BATTERY_STATUS':
            pass

        #print("%s" % name)
#              print msg['current_consumed']
           #   print (msg)

        #print >>sys.stderr, 'sending acknowledgement to', address
        sock.sendto('ack', address)

# MAIN
print("mavsim test agent")
#thread.start_new_thread(mcast_listening_thread,
#                            (mcast_multicast_group, mcast_server_address, ))


#msg = "('LOG', 'CUSTOM_LOG_ENTRY', 'example_agent', 'PERSONAL_LOG', 'I love to fly')"
#msg = "('SIM', 'RESET')"
msg_takeoff = "('FLIGHT', 'AUTO_TAKEOFF')"
#msg = "('FLIGHT', 'AUTO_LAND')"
msg = "('SIM', 'pause')"
print ('Sending takeoff command')
send_message(udp_server_address, msg_takeoff)
print ('Sleepig ...')
time.sleep(5)
print ('Pause')
send_message(udp_server_address, msg)


# while(1):
#     pass




