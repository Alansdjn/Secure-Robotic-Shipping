# -*-coding:utf-8 -*-
# python 2.7
import os
import uuid
import hashlib 
import socket
import rsa
from Crypto.Cipher import AES
import time

address = ('192.168.43.97',33334)# server ip:192.168.43.97
k='d26a53750bc40b38b65a520292f69306'
tid='7db33e3e8dba11ea907b54ee75d57ea6'
ids='c2add694bf942dc77b376592d9c862cd'

#AES encryption/decryption
#Acknowledgement for WangBenYan. Available at:https://blog.csdn.net/qq_18808965/article/details/90262113
class ASEUtil(object): 
    @staticmethod
    def encrypt(key, text):
        bs = AES.block_size
        def pad(s): return s + (bs - len(s) % bs) * chr(bs - len(s) % bs)
        cipher = AES.new(key, AES.MODE_ECB)
        return cipher.encrypt(pad(text)).encode("hex")
    @staticmethod
    def decrypted(key, Enc_Delta):
        cipher = AES.new(key, AES.MODE_ECB)
        def un_pad(s): return s[0:-ord(s[-1])]
        return un_pad(cipher.decrypt(Enc_Delta.decode("hex")))

def MD5(s1):
    md = hashlib.md5()
    md.update(s1)
    return md.hexdigest()

#Generate PK(Public key),PR(Private key)
def create_keys(UAID):  
    (pubkey, privkey) = rsa.newkeys(1024)
    pub = pubkey.save_pkcs1()
    with open('public.pem','wb+')as f:
        f.write(pub) 
    pri = privkey.save_pkcs1()
    pri = pri+UAID
    with open('private.pem','wb+')as f:
        f.write(pri)

#Server-client communication
def socket_sc():

    #Waiting for client's connection
    global address,k,tid,ids
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.bind(address)
    s.listen(5)
    print('Waiting connect...')
    tcpCliSock, addr = s.accept()
    print('connected from client')
    ma1 = tcpCliSock.recv(1024)

    #Confirm client
    if len(ma1)>0: 
        #print('recv ma1:',ma1)
        ma1 = ma1.split(',')
        v0 = ma1[0] 
        n_x = ma1[1]
        tid_r = ma1[2]
        if tid_r == tid:#Check client's ID
            k_r = k
            n_c = ''.join(chr(ord(a)^ord(b)) for a,b in zip(n_x,k_r))
            v0_cal_temp = n_c+tid_r+k_r
            v0_cal = MD5(v0_cal_temp)
            if v0_cal == v0: #Check client's v0
                print('verified client')

                #Generate new values
                tid_new = ''.join(str(uuid.uuid1()).split("-")).upper() #Set new tid
                tid = tid_new 
                n_s=''.join(str(uuid.uuid4()).split("-")).upper() #Random string n_s
                tts = '86400000' #Valid period(24 hour) for QRcode 
                UAID = ''.join(str(uuid.uuid1()).split("-")).upper()#UAID(index) for PK/PR
                create_keys(UAID) #{PK,PR}
                with open('public.pem', 'rb') as publickfile:
                    p = publickfile.read()
                pubkey_temp = p

                #For ma2 to client
                X_temp = n_s+tts+tid_new+UAID+pubkey_temp
                X = ASEUtil.encrypt(k_r,X_temp) 
                R_temp = X+k_r+ids+n_c
                R = MD5(R_temp)
                ma2 = X+','+R
                #print('send ma2:',ma2)	
                tcpCliSock.send(ma2) 

                #For ma3 to robot
                QRcode_temp = n_s +tts + k_r
                QRcode = MD5(QRcode_temp)
                tts_start = int(time.time()) #QRcode's valid period start 
                Enc_Delta_temp = QRcode + ids + str(tts) + str(tts_start) 
                #print('step3_QRcode_delta:',QRcode)
                pubkey = rsa.PublicKey.load_pkcs1(pubkey_temp)#read public key
                Enc_Delta = rsa.encrypt(Enc_Delta_temp, pubkey)
            else:
                print('error connection')	
        else:
            print('error connection')	
    else:
        print('error connection')	
    tcpCliSock.close()
    s.shutdown(socket.SHUT_RDWR)
    s.close()
    return Enc_Delta

#Server-robot communication
def socket_sr(Enc_Delta):
    addr = ('192.168.43.74',33334)# robot ip:192.168.43.74
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.settimeout(10)
    s.connect(addr)
    ma3 = Enc_Delta
    #print('send ma3:',ma3)
    s.sendall(ma3)
    s.shutdown(socket.SHUT_RDWR)
    s.close

if __name__ == '__main__':
    while(1):
        Enc_Delta = socket_sc()
        os.system('sshpass -p ubuntu scp /home/prome/Desktop/project/private.pem ubuntu@192.168.43.74:/home/ubuntu/Desktop/project/')#Assumption of secure channal transferring.
        socket_sr(Enc_Delta)
        raw_input()
