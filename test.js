
var enkf = require('./enkf');

var n = 5, N = 10, m = 1;
var ens = enkf.randn(N, n);
var obs = enkf.randn(1, m);
var H = enkf.obsFunction(1);
var err = [[]], i;
for (i=0; i < m; i++) {
    err[0].push(0*(i+1)/10);
}

var Xa = enkf(ens, obs, H, err);
var pp = enkf.numeric.prettyPrint;

console.log('forecast');
console.log(pp(ens));

console.log('observation');
console.log(pp(obs));

console.log('obs error');
console.log(pp(err));

console.log('analysis');
console.log(pp(Xa));
