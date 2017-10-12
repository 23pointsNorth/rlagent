#!/usr/bin/python

import thread
import time
import socket
import struct
import sys
import future
import yaml
import numpy as np

import map as m

class MAVsimRL:
    mcast_multicast_group = '239.1.1.1'
    mcast_server_address = ('', 10000)
    udp_server_address = ('127.0.0.1', 14555)

    action_wait_time_s = 0.5

    def __init__(self):
        # thread.start_new_thread(self.mcast_listening_thread, (self, ))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print 'Agent initialized!'
        self.takeoff()
        self.pause()

    def __enter__(self):
        return self    

    def __exit__(self, exc_type, exc_value, traceback):
        #self.shutdown_sim()
        self.unpause()
        self.land()
        self.sock.close()
        print 'Closing simulation. Please start sim_vehicle manually.'
        # self.__del__()
    # def __del__(self):

    def reset(self):
        self.send_message("('SIM', 'RESET')")
        print 'Simulation reset'

    def start(self):
        self.send_message("('SIM', 'START')")

    def stop(self):
        self.send_message("('SIM', 'STOP')")

    def shutdown_sim(self):
        self.send_message("('SIM', 'SHUTDOWN')")

    def takeoff(self):
        self.send_message("('FLIGHT', 'AUTO_TAKEOFF')")
        time.sleep(4)
        print 'Plane is in the air'
    def land(self):
        self.send_message("('FLIGHT', 'AUTO_LAND')")
        time.sleep(4)
        print 'Plane is landing ...'

    def pause(self):
        self.send_message("('SIM', 'PAUSE')")

    def unpause(self):
        self.pause()

    def send_message(self, message):
        # Send data
        sent = self.sock.sendto(message, self.udp_server_address)
        # Receive response
        data, server = self.sock.recvfrom(4096)


    def mcast_listening_thread(self):
        # Create the socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind to the server address
        sock.bind(self.mcast_server_address)

        # on all interfaces.
        group = socket.inet_aton(self.mcast_multicast_group)
        mreq = struct.pack('4sL', group, socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        # Receive/respond loop
        while True:
            data, address = sock.recvfrom(1024)

            # Process specific messages
            name = str(data).split(None, 1)[0]
            core = str(data).split(None, 1)[1]
            msg = yaml.load(core)

            # Look for specific messages
            # if name == 'BATTERY_STATUS'
            #     print("%s" % name)
            #     print msg['current_consumed']
            #     print (msg)

    def step(self, action_data=None):
        '''
        Makes a step in the simulation by using the action.
         0 - Nothing
         1 - Set orientation
         2 - Set waypoint
         3 - Takeoff
         4 - Land

        @agent_data - a list of parameters containg the action data. Value '0'
                      specifies the action. The rest, parameters of the action.
        '''
        action = action_data[0]

        # Unpause the simulation
        self.unpause()

        # Remember current state
        start_state = None

        # Make action happen
        self.send_message("('FLIGHT', 'SET_MODE', 'GUIDED')") # GUIDED mode
        self.send_message("('FLIGHT', 'CLEAR_MISSION')")
        if action == 0:
            print 'Doing nothing.'
        elif action == 1:
            heading = np.random.uniform(0, 360)
            distance = 10000.0
            altitude = 100.0
            self.send_message("('FLIGHT', 'HEAD_TO', {0}, {1}, {2})".format(
                                                heading, distance, altitude))
        elif action == 2:
            lat = 0.0
            lon = 35.0 
            absolute_altitude = 900
            self.send_message("('FLIGHT', 'FLY_TO', {0}, {1}, {2})".format(
                                                lat, lon, absolute_altitude))
        elif action == 3:
            self.takeoff()
        elif action == 4:
            self.land()
        else:
            print "UNKNOWN ACTION !!!"

        # Wait a delta t
        time.sleep(self.action_wait_time_s)

        # Get last state
        end_state = None

        # Pause simulation
        self.pause()

        return self.compute_reward(start_state, end_state)
    
    def compute_reward(self, s1, s2):
        return 0


def draw_world(centre):
    height = 256
    width = 256
    scale = 1e-3
    view_map = np.zeros((height,width,3), np.uint8)

    for y in xrange(0, height):
        for x in xrange(0, width):
            if (((y * scale + centre[1])*100)%3 == 0) and (((x * scale + centre[0])*100) %3 == 0):
                view_map[x:y]=256
    cv2.imshow("Map", view_map)
    cv2.waitKey(0)
        

if __name__ == '__main__':
    with MAVsimRL() as agent:
        print("Agent Running ...")
        for x in xrange(1,10):
            agent.step(action_data=[1])
            print 'Step !'




