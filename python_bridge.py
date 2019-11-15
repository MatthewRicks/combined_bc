import os
import sys
import socket
import time
import signal

class PythonBridge:

    def __init__(self, listen_to, send_to, data_size, order = 0):

        self.listen_port = listen_to
        self.send_port = send_to
        self.data_size = data_size
        self.order = order

        self._setup()
        self._install_handle()

    def _install_handle(self):

        def sigint(*args):
            if self.send_socket:
                self.send_socket.shutdown(1)
                self.send_socket.close()
            if self.listen_socket:
                self.client.close()
                self.listen_socket.close()

        signal.signal(signal.SIGINT, sigint)

    def _setup_listen(self):
        self.listen_socket = None
        while not self.listen_socket:
            try:
                self.listen_socket = socket.socket()
                print(('localhost', self.listen_port))
                self.listen_socket.bind(('', self.listen_port))
                self.listen_socket.listen(1)

                self.client, _ = self.listen_socket.accept()
                print('Connected listen_socket')
            except Exception as e:
                print(e)
                self.listen_socket = None
                time.sleep(0.1)
    def _setup_send(self):
        self.send_socket = None
        while not self.send_socket:
            try:
                self.send_socket = socket.socket()
                print(('localhost', self.send_port))
                self.send_socket.connect(('', self.send_port))
                print('Connetected send_socket')
            except Exception as e:
                print(e)
                self.send_socket = None
                time.sleep(0.1)

    def _setup(self):
        if sys.version_info.major == 2 or self.order == 1:
            self._setup_listen()
            self._setup_send()
        else:
            self._setup_send()
            self._setup_listen()

    def send(self, to_send):
        self.send_socket.sendall(to_send)

    def recieve(self):
        recv = self.client.recv(self.data_size)
        return recv


if __name__ == '__main__':

    def py3main():
        conn = PythonBridge(7002, 7001, 13)

        while True:
            s = conn.recieve()
            print(time.time() - float(str(s.decode('utf-8'))))
            conn.send(bytes(str(time.time()), 'utf-8'))

    def py27main():
        conn = PythonBridge(7002, 7001, 18)

        while True:
            conn.send(bytes(str(time.time())))

            s = conn.recieve()
            print(time.time() - float(str(s)))
            #time.sleep(1)

    if 3 == sys.version_info.major:
        print('python 3')
        py3main()
    else:
        py27main()
