#ifndef OTJ_ERROR_HEADER
#define OTJ_ERROR_HEADER 1
void OTJ_Calculate_Error(OTJ_Grid error, OTJ_Grid analytic, OTJ_Grid numerical);
double OTJ_Global_Error(OTJ_Grid error);
double OTJ_Total_Error(OTJ_Grid error);
#endif
