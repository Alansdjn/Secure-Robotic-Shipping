# -*-coding:utf-8 -*-
# python 2.7
import socket
import rsa
import time
import os
from PIL import Image
import pyzbar.pyzbar as pyzbar

address = ('192.168.43.74',33334)# robot ip:192.168.43.74
ids='c2add694bf942dc77b376592d9c862cd'

#Decode QRcode from client
def decode(path):
    frame=Image.open(path)
    barcodes=pyzbar.decode(frame)
    if barcodes:
        print("get QRcode")
        data=''
        for barcode in barcodes:
            data += barcode.data.decode("utf-8")
        #print(data)
        return 1,data
    else:
        print("QRcode scan failed")
        return 0,0

#Reveice messages from server
def socket_server():
    global address,tid
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind(address)
    s.listen(5)
    print('Waiting connect...')
    while(1):
        tcpCliSock, addr = s.accept()
        if addr[0] == '192.168.43.97':
            print('connected from server')
            break
        else:
            print('error')
    ma3 = tcpCliSock.recv(1024)#receive content
    #print('receive ma3:',ma3)	
    tcpCliSock.close()
    s.shutdown(socket.SHUT_RDWR)
    s.close()
    return ma3

#Scan QRcode from client and check it
def QRscan_compare(ma3):
    print('Waiting QRcode...')
    max_time=2
    for t in range(max_time): 
        for i in range(5):
            commands="raspistill -v -o "+str(i)+'.jpg'
            os.system(commands)
            path=str(i)+'.jpg'
            flag,data=decode(path)
            os.remove(path)
            if flag==1:
                break
            elif i==4:
                print('Shift to noncooperative mode.')
                raw_input()#After any input as a signal, shift back to scan QRcode                 
        if flag==1:
            break#Successful scanning
        elif t<max_time-1:
            continue#Shift back from noncooperative mode to scanning QRcode
        else:
            print('Delivery failed.')
            return
    ma4 = data.encode('raw_unicode_escape')#receive content

    #Read private key
    with open('private.pem','rb') as f:  
        p = f.read()
    UAID = p[-32:]
    p = p[:-32]
    privkey = rsa.PrivateKey.load_pkcs1(p)

    #Match messages from server and client
    if len(ma4)>0: 
        #print('recv ma4:',ma4)
        Enc_delta_c = ma4[:-32]
        UAID_c = ma4[-32:]
    if UAID_c == UAID:
        delta_c  = rsa.decrypt(Enc_delta_c, privkey).decode()
        delta_s_temp  = rsa.decrypt(ma3, privkey).decode()
            delta_s = delta_s_temp[:32]
            ids_check = delta_s_temp[32:64]
            tts = int(delta_s_temp[64:72])
            tts_start = int(delta_s_temp[72:])
            tts_end = time.time()
            if tts_end < tts_start + tts:
                if delta_c == delta_s:
                    print("matched!")
    return

if __name__ == '__main__':
    while(1):
        ma3 = socket_server()
        print("please input Y when robot arrives:")
        arrive_flag = raw_input()
        QRscan_compare(ma3)
        raw_input()
