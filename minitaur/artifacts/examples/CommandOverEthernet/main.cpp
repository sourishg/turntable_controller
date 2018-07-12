/*
MIT License (modified)

Copyright (c) 2018 Ghost Robotics
Author: Avik De <avik@ghostrobotics.io>

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
 */
#include <stdio.h>
#include <SDK.h>
#include <Motor.h>
#include <unistd.h>

#pragma pack(push, 1)
// Custom data to append to ethernet packet. No more than GRM_USER_DATA_SIZE=32 bytes!
struct UserData
{
	uint8_t bytes[GRM_USER_DATA_SIZE];
};
#pragma pack(pop)

uint32_t ethRxUpdated;
bool prevJoyRCEnabled = false;
bool joyRCEnabled = true;

class CommandOverEthernet : public Peripheral
{
public:
	void begin()
	{
	}

	// Our received behaviorCmd and user data
	BehaviorCmd behaviorCmd;
	UserData data;

	void update()
	{
		// If no data, stop the robot
		if(S->millis > ethRxUpdated + 2000) // If it's too long past last time we received a packet
		{
			// Stop and re-enable the joystick
			if(!joyRCEnabled)
			{
				C->behavior.mode = 0;
				C->behavior.twist.linear.x = 0;
				joyRCEnabled = true;
			}
		}
		else
		{
			// Read BehaviorCmd from ethernet
			int numRead = read(ETH_UPSTREAM_FILENO, &behaviorCmd, sizeof(BehaviorCmd));
			if(numRead == sizeof(BehaviorCmd))
			{
				// If sensible command
				if(behaviorCmd.id < 10 && behaviorCmd.mode < 10 && 
				   behaviorCmd.twist.linear.x <= 1.0 && behaviorCmd.twist.linear.x >= -1.0 &&
				   behaviorCmd.twist.angular.z <= 1.0 && behaviorCmd.twist.angular.z >= -1.0)
				{
					// Copy to behavior
					memcpy(&C->behavior, &behaviorCmd, sizeof(BehaviorCmd));

					// Disable RC joystick
					joyRCEnabled = false;
				}
			}
		}

		// Disable or enable joystick once
		if(joyRCEnabled && !prevJoyRCEnabled)
		{
			// Enable joystick input
			JoyType joyType = JoyType_FRSKY_XSR;
			ioctl(JOYSTICK_FILENO, IOCTL_CMD_JOYSTICK_SET_TYPE, &joyType);
			printf("Enabling RC Joystick\n");
			C->mode = RobotCommand_Mode_LIMB;
		}
		if(!joyRCEnabled && prevJoyRCEnabled)
		{
			// Disable joystick input
			JoyType joyType = JoyType_NONE;
			ioctl(JOYSTICK_FILENO, IOCTL_CMD_JOYSTICK_SET_TYPE, &joyType);
			printf("Disabling RC Joystick\n");
			C->mode = RobotCommand_Mode_BEHAVIOR;
		}
		prevJoyRCEnabled = joyRCEnabled;

		// Create example user bytes to send back to computer
		for (int i = 0; i < GRM_USER_DATA_SIZE; ++i)
		{
			data.bytes[i] = i;
		}
		write(ETH_UPSTREAM_FILENO, &data, sizeof(UserData));

		// Send state happens automatically
	}
};

CommandOverEthernet commandRobot;

// You can optionally send back a fully custom state packet by replacing the state copy callback.
// If you do this, don't write user bytes using the example user bytes above.
uint16_t myStateCopyCallback(GRMHeader *hdr, uint8_t *buf)
{
	hdr->version = 123;
	for (int i=0; i<10; ++i)
		buf[i] = i;
	return 10;
}

class JointModeStand : public Behavior
{
	void begin() {}
	bool running() { return false; }

	void update()
	{
#if defined(ARM_MATH_CM4) || defined(ARCH_obc)
		// TURN OFF IF REMOTE SWITCH MOVED TO OFF
		if (C->behavior.mode == 0)
		{
			C->mode = RobotCommand_Mode_JOINT;
			for (int i = 0; i < P->joints_count; ++i)
			{
				C->joints[i].mode = JointMode_OFF;
			}
			return;
		}
#endif
		C->mode = RobotCommand_Mode_JOINT;
		// hip/knee
		for (int j = 0; j < 8; ++j)
		{
			joint[j].setOpenLoopMode(jointCurrentControlAvailable ? JointMode_CURRENT : JointMode_PWM);
			if (jointCurrentControlAvailable)
				joint[j].setGain(150, 10);
			else
				joint[j].setGain(0.3);
			joint[j].setPosition(map(/* S->joy.axes[2] */ C->behavior.pose.position.z, -1, 1, 0.4, 1.2));
		}
		// abduction
		for (int j = 8; j < 12; ++j)
		{
			// Can be separate
			joint[j].setOpenLoopMode(jointCurrentControlAvailable ? JointMode_CURRENT : JointMode_PWM);
			// Kppwm * Vbus / (R * Kt) ~= Kppwm * 50
			if (jointCurrentControlAvailable)
				joint[j].setGain(50, 5);
			else
				joint[j].setGain(2.0, 0.005);
			joint[j].setPosition(0);
		}
	}
};

void debug()
{
	// Look at the last time we received from ethernet
	ioctl(ETH_UPSTREAM_FILENO, IOCTL_CMD_GET_LAST_UPDATE_TIME, &ethRxUpdated);
	//printf("%lu\t%d\t%d\n", ethRxUpdated, commandRobot.behaviorCmd.id, commandRobot.behaviorCmd.mode);
}

int main(int argc, char *argv[])
{
#if defined(ROBOT_NGR)
	init(RobotParams_Type_NGR, argc, argv);
	// First NGR batch had directions reversed
	for (int i = 0; i < 8; ++i)
		P->joints[i].direction = -P->joints[i].direction;
	// for debugging init
	JointModeStand jointModeStand;
	behaviors.insert(behaviors.begin(), &jointModeStand);
#else
	// Only for Minitaur E
	init(RobotParams_Type_MINITAUR_E, argc, argv);

	// Remove bound behavior from Minitaur (first element of behaviors vector),
	// so we're left with only walk behavior
	behaviors.erase(behaviors.begin());
#endif
	// Set joystick
	JoyType joyType = JoyType_FRSKY_XSR;
	ioctl(JOYSTICK_FILENO, IOCTL_CMD_JOYSTICK_SET_TYPE, &joyType);

	// Create controller peripheral
	commandRobot.begin();

	// Add it
	addPeripheral(&commandRobot);

	// Replace state copy callback
	//ioctl(ETH_UPSTREAM_FILENO, IOCTL_CMD_STATE_COPY_CALLBACK, (void *)myStateCopyCallback);

	// Set debug rate
	setDebugRate(10);

	// Go
	return begin();
}
