
fun = @(phi) ((phi(1)*phi(2))^2)/(4/pi);


%initial unoptimised parameters - spiral angle, pitch, pitch at max
%deflection, thickness of spiral
phi_0 = [32.08,3.618,5.16,2.12];
lb = [];
ub = [];
A = [];
b1 = [];
Aeq = [];
beq = [];
% performing optimisation
x = fmincon(fun,phi_0,A,b1,Aeq,beq,lb,ub,@spring_opti)