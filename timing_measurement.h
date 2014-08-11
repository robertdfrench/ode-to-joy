#ifndef OTJ_TIMING_MEASUREMENT
#define OTJ_TIMING_MEASUREMENT 1
typedef struct timing_measurement_t {
	double beginning;
	double end;
	char* message;
} TimingMeasurement;

void stop_timer(TimingMeasurement tm);
TimingMeasurement start_timer(char* message);
#endif
