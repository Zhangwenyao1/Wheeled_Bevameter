import math
import time
import numpy as np
import struct
import socket
import os
import threading    
# import msvcrt
from dynamixel_sdk import * 
# import sinkage_detect
import detect_sinkage
import detectsink
import xlsxwriter
import matplotlib.pyplot as plt
# from tests.protocol2_0.detect_sinkage import detect

class Press_experiment:
    ADDR_PRO_TORQUE_ENABLE = 64  # Control table address is different in Dynamixel model
    ADDR_PRO_GOAL_POSITION = 116
    ADDR_PRO_PRESENT_POSITION = 132
    ADDR_PRO_PRESENT_CURRENT = 126
    ADDR_PRO_PROFILE_VELOCITY = 112
    ADDR_PRO_GOAL_VELOCITY = 104
    LEN_PRO_PROFILE_VELOCITY = 4
    ADDR_PRO_OPERATING_MODE = 11
    ADDR_PRO_Return_Delay_Time  = 9
    ADDR_PRO_GOAL_CURRENT = 102


    Current_Model = 0
    Velocity_Model = 1
    Position_Model = 3
    # Protocol version
    PROTOCOL_VERSION = 2.0  # See which protocol version is used in the Dynamixel

    # Default setting
    DXL0_ID = 0
    DXL1_ID = 1  # Dynamixel ID : 1
    DXL2_ID = 2  # Dynamixel ID : 2
    DXL3_ID = 3  # Dynamixel ID : 3
    BAUDRATE = 1000000  # Dynamixel default baudrate : 57600
    DEVICENAME = '/dev/ttyUSB0'  # Check which port is being used on your controller
    # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

    TORQUE_ENABLE = 1  # Value for enabling the torque
    TORQUE_DISABLE = 0  # Value for disabling the torque

    DXL_MOVING_STATUS_THRESHOLD = 5  # Dynamixel moving status threshold
    PROFILE_VELOCITY = 5

    def __init__(self):
        super().__init__()
        pass
        # Control table address
       
    def output(self):
        self.portHandler = PortHandler(self.DEVICENAME)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        return self.packetHandler,self.portHandler


    def getch (self) :
        return msvcrt.getch( ).decode( )

    def InitDevice(self):
        # portHandler = PortHandler(self.DEVICENAME)

        # # Initialize PacketHandler instance
        # # Set the protocol version
        # # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        # packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # Open port
        if self.portHandler.openPort( ) :
            print("-----------")
        else :
            self.getch( )
            quit( )
            print("-----------")
            # print("Failed to open the port")
            # print("Press any key to terminate...")
        # Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE) :
            print("-----------")
            # print("Succeeded to change the baudrate")
        else :
            print("------")
            # print("Press any key to terminate...")
            self.getch( )
            quit( )

        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL0_ID, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_DISABLE)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL1_ID, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_DISABLE)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL2_ID, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_DISABLE)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL3_ID, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_DISABLE)
        # set return delay time
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL0_ID,self.ADDR_PRO_Return_Delay_Time, 0)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL1_ID, self.ADDR_PRO_Return_Delay_Time, 0)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL2_ID, self.ADDR_PRO_Return_Delay_Time, 0)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL3_ID, self.ADDR_PRO_Return_Delay_Time, 0)



        # print("Dynamixel has been successfully connected")
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL0_ID, self.ADDR_PRO_OPERATING_MODE, self.Position_Model)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL1_ID, self.ADDR_PRO_OPERATING_MODE, self.Position_Model)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL2_ID, self.ADDR_PRO_OPERATING_MODE, self.Position_Model)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL3_ID, self.ADDR_PRO_OPERATING_MODE, self.Velocity_Model)
        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL0_ID, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_ENABLE)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL1_ID, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_ENABLE)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL2_ID, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_ENABLE)
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL3_ID, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_ENABLE)
        # # # if dxl_comm_result != COMM_SUCCESS :
        #     # print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        # elif dxl_error != 0 :
        #     # print("%s" % packetHandler.getRxPacketError(dxl_error))
        # else :

        # dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL3_ID, ADDR_PRO_OPERATING_MODE, Velocity_Model)

        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL0_ID, self.ADDR_PRO_PROFILE_VELOCITY,
                                                                20)
        dxl_comm_result, dxl_error =self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL1_ID, self.ADDR_PRO_PROFILE_VELOCITY,
                                                                20)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL2_ID, self.ADDR_PRO_PROFILE_VELOCITY,20)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL3_ID, self.ADDR_PRO_GOAL_VELOCITY,GOAL_VELOCITY)
        




        #---------------------??????????????????---------------------------------------------
        IP_ADDR = '192.168.0.108'
        PORT = 4008
        #??????????????????
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s = s
        #??????

        s.connect((IP_ADDR, PORT))
        #??????????????????
        sendData = "AT+EIP=?\r\n"
        s.send(sendData.encode())
        recvData = bytearray(s.recv(1000))
        # print(recvData.decode())

        #??????????????????
        sendData = "AT+DCPM=?\r\n"
        s.send(sendData.encode())
        recvData = bytearray(s.recv(1000))
        # print (recvData.decode())

        #??????????????????
        sendData = "AT+DCPCU=?\r\n"
        s.send(sendData.encode())
        recvData = bytearray(s.recv(1000))
        # print (recvData.decode())

        #?????????????????????
        #??????????????????,?????????????????????
        # decouple_matrix = "(0.74447,-0.10432,-4.05718,113.20804,-0.19615,-112.78654);" \
        #                       "(-1.55915,-131.20568,-12.28236,63.88653,2.47576,65.56931);" \
        #                       "(320.82506,-6.00990,322.04951,1.83496,324.31629,-3.44262);" \
        #                       "(0.13401,0.15065,-11.66469,-0.15551,11.42006,-0.23770);" \
        #                       "(13.19140,-0.21842,-6.52051,0.03898,-6.59893,0.00286);" \
        #                       "(-0.01730,4.59975,-0.19084,4.56898,0.06837,4.54050)\r\n";
        # set_decouple_matrix = "AT+DCPM=" + decouple_matrix
        # s.send(set_decouple_matrix)
        # recvData = bytearray(s.recv(1000))
        # print "???????????????????????????", recvData


        #??????????????????
        set_update_rate = "AT+SMPR=100\r\n"
        s.send(set_update_rate.encode())
        recvData = bytearray(s.recv(1000))
        # print (recvData.decode())

        #????????????????????????
        set_compute_unit = "AT+DCPCU=MVPV\r\n"
        s.send(set_compute_unit.encode())
        recvData = bytearray(s.recv(1000))
        # print (recvData.decode())

        #??????????????????
        set_recieve_format= "AT+SGDM=(A01,A02,A03,A04,A05,A06);E;1;(WMA:1)\r\n"
        s.send(set_recieve_format.encode())
        recvData = bytearray(s.recv(1000))
        # print (recvData.decode())
            #?????????????????????
        get_data_stream = "AT+GSD\r\n"
        s.send(get_data_stream.encode())
        self.s = s

    def InverseKinematics(self,x,y):
        L1 = 1.054
        L2 = 1
        beta = math.acos((L1**2 + L2 **2 - x**2 - y**2)/(2*L1*L2))
        alpha = math.acos((x**2 + y**2 + L1**2 - L2**2)/(2*L1*((x**2 + y**2)**0.5)))
        gama = math.atan2(y, x)
        theta1 = gama + alpha
        theta2 = beta - math.pi
        return theta1, theta2

    def motor_move(self,motor1,motor2):
        # portHandler = PortHandler(self.DEVICENAME)

        # # Initialize PacketHandler instance
        # # Set the protocol version
        # # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        # packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler,self.DXL0_ID, self.ADDR_PRO_GOAL_POSITION,int((math.pi-motor1-0.2548)/2/math.pi*4096))
        # dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL1_ID, ADDR_PRO_GOAL_POSITION,int((math.pi-motor1)/2/math.pi*4096))
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, self.DXL2_ID, self.ADDR_PRO_GOAL_POSITION,int((motor2+1.5*math.pi-0.2548)/2/math.pi*4096))
        # Read present position
        dxl_present_position0, dxl_comm_result, dxl_error =packetHandler.read4ByteTxRx(self.portHandler, self.DXL0_ID,
                                                                                        self.ADDR_PRO_PRESENT_POSITION)
        # print("ID0:{0}".format(dxl_present_position0))

        dxl_present_position1, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(self.portHandler, self.DXL1_ID,
                                                                                        self.ADDR_PRO_PRESENT_POSITION)
        # print("ID1:{0}".format(dxl_present_position1))
        dxl_present_position2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(self.portHandler, self.DXL2_ID,
                                                                                        self.ADDR_PRO_PRESENT_POSITION)
        # print("ID2:{0}".format(dxl_present_position2))
        if dxl_comm_result != COMM_SUCCESS :
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0 :
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))


    def move(self,x,y):
        motor1, motor2 = self.InverseKinematics(x,y)
        self.motor_move(motor1, motor2)
        return motor1, motor2


    # def forwardkinematics(theta1,theta2):
    #     L1 = 1
    #     L2 = 2
    #     x = L1*math.cos(theta1) + L2 * math.cos(theta1+theta2)
    #     y = L1*math.sin(theta1) + L2 * math.sin(theta1+theta2)


    def getForce(self):
        

        data = self.s.recv(1000)
        fx = struct.unpack("f", data[6:10])[0]
        fy = struct.unpack('f', data[10:14])[0]
        fz = struct.unpack('f', data[14:18])[0]
        mx = struct.unpack('f', data[18:22])[0]
        my = struct.unpack('f', data[22:26])[0]
        mz = struct.unpack('f', data[26:30])[0]
        F = np.array([fx, fy, fz, mx, my, mz])
        # print (np.around(F, 4))x
        # print(np.around(fx, 4))
        #????????????
        # stop_data_stream = "AT+GSD=STOP\r\n"
        # s.send(stop_data_stream.encode())
        # recvData = bytearray(s.recv(1000))
        # print (recvData.decode())
        # print('fx',abs(fx-5.8))
        # print('fz',fz)
        # print('pullforce',mz)
        # print('mx',mx)
        # print('my',my)
        print('fy',fy-1)
        return abs(fx),fy,fz,mx,my,mz


        # Close port
        portHandler.closePort( )

def normal_force(dxl_present_position0 ,dxl_present_position2,fx,fy):
    theta1 = math.pi-dxl_present_position0/4096*2*math.pi 
    theta2 = dxl_present_position2/4096*2*math.pi - math.pi*1.5
    theta3 = math.pi/2+theta2
    theta4 = -(theta1 +theta3)
    rotate_matrix = np.zeros((3,3))
    rotate_matrix[0,:] = [math.cos(theta4),-math.sin(theta4),0]
    rotate_matrix[1,:] = [math.sin(theta4),math.cos(theta4),0]

    rotate_matrix[2,:] = [0,0,1]
    force_matrix =np.zeros((3,1))
    force_matrix = [fy,fx,fz]

    force_matrix = np.dot(rotate_matrix,force_matrix)
    normal_force1= force_matrix[1]
    drawbar_pull1 = force_matrix[0]
    theta4 = theta1 +theta3
    # print('theta4',theta4)
    if theta4<0:
        theta4 = -theta4
        normal_force = abs(abs(fx)*math.cos(theta4) + abs(fy)*math.sin(theta4))
        drawbar_pull = abs(abs(fx)*math.sin(theta4) - abs(fy)*math.cos(theta4))
    else:
        normal_force = abs(abs(fx)*math.cos(theta4) + abs(fy)*math.sin(theta4))
        drawbar_pull = abs(abs(fy)*math.cos(theta4)-abs(fx)*math.sin(theta4))
    # # print('------------------------')
    # # normal_force = abs(abs(fx)*math.cos(theta4) + abs(fy)*math.sin(theta4))
    # # print('theta1',theta1)
    # # print('theta2',theta2)

    # print('fx',fx)
    # if theta4 > 0.01:
    #     drawbar_pull = 0
    
    # print('fy',fy)
    # print('fz',fz)
    # # # fy = abs(fy)-math.sin(abs(theta4))*2.5
    # # # print('------------------------')
    print('drawbar_pull',drawbar_pull1)
    # print(fy)
    print(drawbar_pull)
    
    print('normal_foece11111',normal_force1)
    # print('normal_force',normal_force)
    # if abs(theta2)<math.pi/2:
    #     normal_force = normal_force + normal_force*0.5
    # normal_force = abs(fx)*math.cos(theta1+theta2)+abs(fy)*math.sin(theta1+theta2)

    return normal_force1,drawbar_pull1










    
        

def jacobin(theta1,theta2):
    L1 = 1.054
    L2 = 1

    jacobin_matrix = np.zeros((2,2))
    jacobin_matrix[0][:] = [-L1*math.sin(theta1)-L2*math.sin(theta1+theta2),-L2*math.sin(theta1+theta2)]
    jacobin_matrix[1][:] = [L1*math.cos(theta1)+L2*math.cos(theta1+theta2),L2*math.cos(theta1+theta2)]
    return jacobin_matrix

# def forward_kinematics(theta1,theta2):   
        

def pidControl(force,last_bias):
    
    # Kp = 0.002255
    Kp = 0.00263
    Kp = 0.00027
    # Kp = -13.1 # df = 30
    # Kp = -12 # df = 35
    # Kp = -10  #df = 25
    # Kp = 0.00045
    Kp = -5
    bias = force - desiredForce
    # if abs(bias) < 8:
    #     balance = bias * Kp
    # else:
    #     balance = 0
    
    Kd = -14 # df = 35
    Kd = -6  # df = 30
    Kd = -6  # df = 25
    # bias = force - desiredForce

    balance = Kp*bias +Kd*(bias-last_bias)

    return balance,bias



def fkinematics(theta1,theta2):
    L1 = 1.054
    L2 = 1
    x = L1*math.cos(theta1)+L2*math.cos(theta1+theta2)
    y = L1*math.sin(theta1)+L2*math.sin(theta1+theta2)
    return x,y

def draw(bias_data):
    new_data = bias_data[100:]
    restore = coordinate_x_restore[100:]
    lenth_restore = []
    # for i in range(len(new_data)):
    #     lenth_restore.append(coordinate_x_restore)
    plt.plot(restore, new_data)
    plt.title('Pull Force')
    plt.xlabel('Time')
    plt.ylabel('Pull Force')
    plt.show()



GOAL_VELOCITY = 20
linspace = 0.00075
linspace = 0.001
# linspace = 0.0005
desiredForce = 20


if __name__ == '__main__':

    if os.name == 'nt' :
        import msvcrt

        def getch ( ) :
            return msvcrt.getch( ).decode( )
    else :
        import sys, tty, termios

        fd = sys.stdin.fileno( )
        old_settings = termios.tcgetattr(fd)


        def getch ( ) :
            try :
                tty.setraw(sys.stdin.fileno( ))
                ch = sys.stdin.read(1)
            finally :
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

    from dynamixel_sdk import *  # Uses Dynamixel SDK library
    A = Press_experiment()
    packetHandler, portHandler = A.output()
    A.InitDevice()
    coordinate_x =0.6
    
    force = 4  
    row = 0   
    bias_restore = []
    fx_restore = []
    fy_restore = []
    fz_restore = []
    pf_restore = []
    dx0crestore =[]
    dx1crestore =[]
    theta4_restore = []
    theta1_restore = []
    theta2_restore = []
    drawbar_pull_restore = []
    coordinate_x_restore = []
    while True:
        _,_=A.move(0.7, 0)     
        print("Press any key to continue! (or press ESC to quit!)")
        if getch() == chr(0x1b):
            break
        item = 0     
        workbook = xlsxwriter.Workbook('1'+'Freeslip_load'+str(desiredForce)+'--'+'.xlsx')
        worksheet = workbook.add_worksheet()  
        while True:
            fx,fy,fz,mx,my,mz = A.getForce()
            dxl_present_position0, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PRESENT_POSITION)
            dxl_present_position2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_PRESENT_POSITION)
            nf,drawbar_pull = normal_force(dxl_present_position0,dxl_present_position2,fx,fy)
            # force = getForce()[0]
            # print('FORCE',force)
            # coordinate_y = coordinate_y + pidControl(force)
            down_velocity =20
            dxl_comm_result, dxl_error =packetHandler.write4ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PROFILE_VELOCITY,
                                                            down_velocity)
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, A.DXL1_ID, A.ADDR_PRO_PROFILE_VELOCITY,
                                                            down_velocity)
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_PROFILE_VELOCITY,
                                                          down_velocity)

            index = 0 
                                                                                                 
            for i in np.arange(0,-1.9,-0.001):
                motor1,motor2 =A.move(0.7, i)
                # motor1, motor2 = InverseKinematics(coordinate_x, i)
                dxl_goal_position0 = int((math.pi-motor1-0.2548)/2/math.pi*4096)
                dxl_goal_position2 = int((motor2+1.5*math.pi-0.2548)/2/math.pi*4096)
                y = i
                while True :
                   
                    fx,fy,fz,mx,my,mz = A.getForce()
                    dxl_present_position0, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PRESENT_POSITION)
                    dxl_present_position2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_PRESENT_POSITION)
                    nf,drawbar_pull = normal_force(dxl_present_position0,dxl_present_position2,fx,fy)     
                    # print(fx)
                    # print(fy)
                    # print(nf)
                    if (abs(dxl_goal_position0 - dxl_present_position0) <40) and (abs(dxl_goal_position2 - dxl_present_position2) <40):
                        break
                if 100>nf >= 10: 
                    index +=1
                if index == 2 :
                    break
            last_bias = 0
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_TORQUE_ENABLE,A.TORQUE_DISABLE)
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_TORQUE_ENABLE,A.TORQUE_DISABLE)
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_OPERATING_MODE, A.Current_Model)
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_TORQUE_ENABLE,A.TORQUE_ENABLE)
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_GOAL_CURRENT,200) # 20N 140  30N 250 10N 50 15N 100 25n 200
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, A.DXL3_ID, A.ADDR_PRO_GOAL_VELOCITY,GOAL_VELOCITY)
            while True:
                dxl_present_position0, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PRESENT_POSITION)
                dxl_present_position2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_PRESENT_POSITION)
                fx,fy,fz,mx,my,mz = A.getForce()
                nf,drawbar_pull = normal_force(dxl_present_position0,dxl_present_position2,fx,fy)
                theta1 = math.pi-dxl_present_position0/4096*2*math.pi 
                theta2 = dxl_present_position2/4096*2*math.pi - math.pi*1.5
                theta3 = math.pi/2+theta2
                theta4 = abs(theta1 +theta3)  
                theta4_restore.append(theta4)

                if 100<nf or nf<-2:
                    continue
                theta1 = math.pi - dxl_present_position0/4096 *2*math.pi - 0.2548 # -0.255?????????????????????????????????
                theta2 = dxl_present_position2/4096 *2*math.pi  - 1.5*math.pi+0.2548
                
                coordinate_x,coordinate_y = fkinematics(theta1,theta2)
                
                # max_x = round((2.045**2 - coordinate_y**2)**0.5,3)-0.2
                max_x = 1.3
                start_x =0.73
                

                # if 0.898< i <= 0.903:
                if coordinate_x < start_x+1*linspace:
                    # start_x = i
                    start_time = time.time()
                    
                deltay,bias = pidControl(nf,last_bias)
                # print('deltay',deltay)
                last_bias = bias
                current,_,_ = packetHandler.read2ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PRESENT_CURRENT)
                current +=deltay
                time_label = time.time()-start_time
                row += 1         
                packetHandler.write2ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_GOAL_CURRENT,int(current))
        #         # motor1, motor2 = InverseKinematics(coordinate_x, i)
                coordinate_x_restore.append(coordinate_x)
                bias_restore.append(nf)
                pull_force = abs(mz/0.066)
                pf_restore.append(pull_force)
                fx_restore.append(fx)
                fy_restore.append(fy)
                drawbar_pull_restore.append(drawbar_pull)
        #         fz_restore.append(fz)
                # #         #     worksheet.write(row, 1, round(time_label,4))  # ???i???0???
                worksheet.write(row, 1,str(nf))  # ???i???0???
                # worksheet.write(row, 3, round(i,4))  # ???i???0???
                worksheet.write(row, 2, fx)
                # worksheet.write(row, 5, fy)
                worksheet.write(row, 3, pull_force)
                # worksheet.write(row, 7, mz)
                # worksheet.write(row, 4, round((dxl_present_current0-16**3*15-16**2*15)*2.69,4))
                # worksheet.write(row, 5, round((dxl_present_current2-16**3*15-16**2*15)*2.69,4))
                if coordinate_x >max_x-linspace:
                    end_time = time.time()
                    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_TORQUE_ENABLE,A.TORQUE_DISABLE)
                    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_OPERATING_MODE, A.Position_Model)
                    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_TORQUE_ENABLE,A.TORQUE_ENABLE)
                    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_TORQUE_ENABLE,A.TORQUE_ENABLE)
                    # dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_GOAL_CURRENT,250)
                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PROFILE_VELOCITY,30)
                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_PROFILE_VELOCITY,30)
                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, A.DXL3_ID, A.ADDR_PRO_GOAL_VELOCITY,0)
                    break
        #         # for i in np.arange(0.79,max_x,0.00083):
        # #         #     if 100<nf or nf<-2:
        # #         #         continue
                    
        # #         #     # if 0.898< i <= 0.903:
        # #         #     # if i == start_x:
        # #         #     #     start_x = i
        # #         #     #     start_time = time.time()
        # #         #     if i >max_x-0.001:
        # #         #         end_time = time.time()
        # #         #     deltay,bias = pidControl(nf,last_bias)
        # #         #     # print('deltay',deltay)
        # #         #     last_bias = bias
        # #         #     coordinate_y +=deltay
        # #         #     time_label = time.time()-start_time
        # #         #     dxl_present_position0, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PRESENT_POSITION)
        # #         #     dxl_present_position2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_PRESENT_POSITION)
        # #         #     dxl_present_current0, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PRESENT_CURRENT)
        # #         #     dxl_present_current2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_PRESENT_CURRENT)
        # #         #     nf = normal_force(dxl_present_position0,dxl_present_position2,fx,fy,fz)
        # #         #     pull_force = abs(mz/0.066)
        # #         #     if pull_force>-7:
        # #         #         a=1
        # #         #     bias_restore.append(nf)
        # #         #     pf_restore.append(pull_force)
        # #         #     fx_restore.append(fx)
        # #         #     fy_restore.append(fy)
        # #         #     fz_restore.append(fz)
        # #         #     print('pull_force',pull_force)
        # #         #     worksheet.write(row, 1, round(time_label,4))  # ???i???0???
        # #         #     worksheet.write(row, 2,str(nf))  # ???i???0???
        # #         #     worksheet.write(row, 3, round(i,4))  # ???i???0???
        # #         #     worksheet.write(row, 4, fx)
        # #         #     worksheet.write(row, 5, fy)
        # #         #     worksheet.write(row, 6, pull_force)
        # #         #     worksheet.write(row, 7, mz)
        # #         #     worksheet.write(row, 4, round((dxl_present_current0-16**3*15-16**2*15)*2.69,4))
        # #         #     worksheet.write(row, 5, round((dxl_present_current2-16**3*15-16**2*15)*2.69,4))
                    
        # #         #     row += 1         
        # #         #     motor1,motor2 =A.move(i, coordinate_y)
        # #         #     # motor1, motor2 = InverseKinematics(coordinate_x, i)
        # #         #     while True :
        # #         #         print('pull_force',pull_force)
        # #         #         dxl_goal_position0 = int((math.pi-motor1-0.2548)/2/math.pi*4096)
        # #         #         dxl_goal_position2 = int((motor2+1.5*math.pi-0.2548)/2/math.pi*4096)
        # #         #         fx,fy,fz,mx,my,mz = A.getForce()
        # #         #         pull_force = abs(mz/0.066)
        # #         #         dxl_present_position0, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL0_ID, A.ADDR_PRO_PRESENT_POSITION)
        # #         #         dxl_present_position2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, A.DXL2_ID, A.ADDR_PRO_PRESENT_POSITION)
        # #         #         nf = normal_force(dxl_present_position0,dxl_present_position2,fx,fy,fz)

                        
        # #         #         # coordinate_y = coordinate_y+deltay
        # #         #         if (abs(dxl_goal_position0 - dxl_present_position0) <70) and (abs(dxl_goal_position2 - dxl_present_position2) <70):
        # #         #             break


                              
                
            break
        workbook.close()
        total_time = end_time - start_time
        velocity = (max_x - start_x)*0.22552/total_time
        omiga = GOAL_VELOCITY*0.229*2*math.pi/60
        slip_ratio = (omiga*0.066-velocity)/(omiga*0.066)
        # pf_restore = pf_restore[100:]
        # print('mean_pull',sum(pf_restore)/len(pf_restore))
        print('slip_ratio',slip_ratio)
        print('total_time',total_time)
        print('velocity',velocity)
        # item += 1
        # worksheet.write(row, 8, slip_ratio)
        # worksheet.write(row, 9, velocity)
        # worksheet.write(row, 10, GOAL_VELOCITY)
        draw(bias_restore)
        draw(pf_restore)
        draw(drawbar_pull_restore)
        draw(fx_restore)
        draw(fy_restore)
        draw(theta4_restore)
                            


