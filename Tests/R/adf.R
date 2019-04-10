# Source: http://fabian-kostadinov.github.io/2015/01/27/comparing-adf-test-functions-in-r/

library(tseries)
library(xts)
library(fUnitRoots)

flat0 <- xts(rnorm(100), Sys.Date()-100:1)
plot(flat0)

flat20 <- xts(rnorm(100), Sys.Date()-100:1)+20
plot(flat20)

trend0 <- flat0+(row(flat0)*0.1)
plot(trend0)

trend20 <- flat0+(row(flat0)*0.1)+20
plot(trend20)

adf.test(flat0, alternative = "stationary", k = 0)
adf.test(flat20, alternative = "stationary", k = 0)
adf.test(trend0, alternative = "stationary", k = 0)
adf.test(trend20, alternative = "stationary", k = 0)

adfTest(flat0, lags = 0, type = "nc")
adfTest(flat20, lags = 0, type = "nc")
adfTest(trend0, lags = 0, type = "nc")
adfTest(trend20, lags = 0, type = "nc")

adfTest(flat0, lags = 0, type = "c")
adfTest(flat20, lags = 0, type = "c")
adfTest(trend0, lags = 0, type = "c")
adfTest(trend20, lags = 0, type = "c")

adfTest(flat0, lags = 0, type = "ct")
adfTest(flat20, lags = 0, type = "ct")
adfTest(trend0, lags = 0, type = "ct")
adfTest(trend20, lags = 0, type = "ct")