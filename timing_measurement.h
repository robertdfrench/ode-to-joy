#ifndef OTJ_TIMING_MEASUREMENT
#define OTJ_TIMING_MEASUREMENT 1
typedef struct OTJ_Timer_t {
	double beginning;
	double end;
	char* message;
} OTJ_Timer;

void OTJ_Timer_Stop(OTJ_Timer tm);
OTJ_Timer OTJ_Timer_Start(char* message);
#endif
