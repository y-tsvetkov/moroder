function l = spiral_length(theta,p)
l = (p/(4*pi))*(theta*sqrt(1+theta^2) + log(theta + sqrt(1+theta^2)));