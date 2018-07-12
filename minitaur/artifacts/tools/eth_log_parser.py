'''
MIT License (modified)

Copyright (c) 2018 Ghost Robotics
Authors:
Avik De <avik@ghostrobotics.io>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this **file** (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
Documentation:
- log data using a custom ethernet state copy callback
- depending on your computer's ethernet driver or switch latency, you might want to log S->millis
- open wireshark to log data (connect ethernet cable to mbm bottom port)
- install `pip install scapy`
- provide log filename as argument
'''

import scapy.all
import struct
import sys
# for data manipulation
import numpy as np
import matplotlib.pyplot as plt

# Header
headerFmt = '<BBHiBB'
headerSize = struct.calcsize(headerFmt)

t = np.zeros((0))

# # Custom data: millis + 2x Imu
# ImuFmt = '9f'
# dataFmt = '<I' + ImuFmt + ImuFmt
# euler1 = np.zeros((0, 3))
# angular_velocity1 = np.zeros((0, 3))
# linear_acceleration1 = np.zeros((0, 3))
# euler2 = np.zeros((0, 3))
# angular_velocity2 = np.zeros((0, 3))
# linear_acceleration2 = np.zeros((0, 3))

# Custom data: millis + Njoints * (raw joint data)
Njoints = 3
LANIfmt = '2f4H'
dataFmt = '<I'
for i in range(Njoints):
	dataFmt = dataFmt + LANIfmt
pos = np.zeros((0, Njoints))
vel = np.zeros((0, Njoints))
cur = np.zeros((0, Njoints))
temperature = np.zeros((0, Njoints))
voltage = np.zeros((0, Njoints))
param4 = np.zeros((0, Njoints))


dataSize = struct.calcsize(dataFmt)
# rdpcap comes from scapy and loads in our pcap file
packets = scapy.all.rdpcap(sys.argv[1])

# Let's iterate through every packet
for packet in packets:
	if packet.haslayer(scapy.all.IP):
		if packet.getlayer(scapy.all.IP).src != '169.254.98.123':
			continue
			
	if packet.haslayer(scapy.all.UDP):
		if packet.getlayer(scapy.all.UDP).dport != 15000:
			continue
		alldata = packet.getlayer(scapy.all.UDP).payload.load
		magic, version, totalSize, headerCrc, numDoF, configBits = struct.unpack(headerFmt, alldata[:headerSize])
		# can check version here
		data = struct.unpack(dataFmt, alldata[headerSize: headerSize + dataSize])

		offset = 0
		t = np.hstack((t, 0.001 * data[offset]))
		offset += 1

		# # custom data: 2x IMU
		# angular_velocity1 = np.vstack((angular_velocity1, data[offset:offset + 3]))
		# offset += 3
		# linear_acceleration1 = np.vstack((linear_acceleration1, data[offset:offset + 3]))
		# offset += 3
		# euler1 = np.vstack((euler1, data[offset:offset + 3]))
		# offset += 3
		# angular_velocity2 = np.vstack((angular_velocity2, data[offset:offset + 3]))
		# offset += 3
		# linear_acceleration2 = np.vstack((linear_acceleration2, data[offset:offset + 3]))
		# offset += 3
		# euler2 = np.vstack((euler2, data[offset:offset + 3]))
		# offset += 3

		# custom data: Njoints * raw joint data
		pos = np.vstack((pos, np.zeros((1, Njoints))))
		vel = np.vstack((vel, np.zeros((1, Njoints))))
		cur = np.vstack((cur, np.zeros((1, Njoints))))
		temperature = np.vstack((temperature, np.zeros((1, Njoints))))
		voltage = np.vstack((voltage, np.zeros((1, Njoints))))
		param4 = np.vstack((param4, np.zeros((1, Njoints))))
		
		for j in range(Njoints):
			pos[-1, j] = data[offset]
			vel[-1, j] = data[offset + 1]
			cur[-1, j] = data[offset + 2]
			temperature[-1, j] = data[offset + 3]
			voltage[-1, j] = 0.001 * data[offset + 4]
			param4[-1, j] = data[offset + 5]
			offset += 6
		
		
print 'Average data rate =', 1.0/np.mean(np.diff(t)), 'Hz'

plt.figure()

# titles = ['Roll', 'Pitch', 'Yaw']
# for i in range(3):
# 	plt.subplot(3,1,i+1)
# 	plt.plot(t, np.unwrap(euler1[:, i]))
# 	plt.plot(t, np.unwrap(euler2[:, i]))
# 	plt.title(titles[i])
# plt.figure()
# titles = ['wx', 'wy', 'wz']
# for i in range(3):
# 	plt.subplot(3, 1, i+1)
# 	plt.plot(t, angular_velocity1[:, i])
# 	plt.plot(t, angular_velocity2[:, i])
# 	plt.title(titles[i])
# plt.figure()
# titles = ['ax', 'ay', 'az']
# for i in range(3):
# 	plt.subplot(3, 1, i+1)
# 	plt.plot(t, linear_acceleration1[:, i])
# 	plt.plot(t, linear_acceleration2[:, i])
# 	plt.title(titles[i])
		
# joints
plt.subplot(3,1,1)
plt.plot(t, pos)
plt.subplot(3, 1, 2)
plt.plot(t, vel)
plt.subplot(3, 1, 3)
plt.plot(t, voltage)


plt.show()
