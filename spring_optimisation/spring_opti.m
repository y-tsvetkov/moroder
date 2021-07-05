
function [c,ceq] = spring_opti(x)

delta = 0.6; %N/A - maximum stress expressed as a factor of material yield stress
n_m = 4; %N/A - number of minimal rounds
sigma_y = 58; %MPa - material yield stress
E = 2100; %MPa - Young's modulus

%input data
theta_m = 2*pi; %rad - maximal desired deflection
b = 2; %thickness of spring normal to the spiral plane
p_0 = 0.5; %mm - initial pitch
R_s = 2; %mm - radius of shaft that doesn't bend
K_d = 245.16625/(4*pi); %Nmm/rad - desired torque

L = @spiral_length;
L_s = L(2*pi*R_s/x(2),x(2));

%implementing equations and inequalities from the paper
c(1) = E*x(4)/(2*(L(x(1),x(2)) - L_s)) - delta*sigma_y;
c(2) = 2*n_m*pi - x(1);

ceq(1) = (E*b*(x(4)^3))/(12*(L(x(1),x(2)) - L_s)) - K_d;
ceq(2) = L(x(1),x(2)) - L(x(1) + theta_m,x(3));
ceq(3) = x(4) + p_0 - x(3);