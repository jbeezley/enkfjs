
var enkf = require('./enkf');

var n = 5, N = 10;
var ens = enkf.randn(N, n);
var obs = enkf.randn(1, n);
var H = enkf.obsFunction();
var err = [[]], i;
for (i=0; i < n; i++) {
    err[0].push((i+1)/10);
}

var Xa = enkf(ens, obs, H, err);
