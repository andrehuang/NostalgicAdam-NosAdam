function [f,g1,g2] = drill(x,y,z)
if nargin<3
    z = 2.34;
end
r = 1.3; beta = sqrt(400); % determine the location and narrowness of the local minima
a = 30;
b = 2;
c = 6;
f = -a*exp(-b*((x-pi)^2 + (y-pi)^2 + (z-pi)^2));
% f = 0;
g1 = -a*exp(-b*((x-pi).^2 + (y-pi).^2 + (z-pi).^2)).*(-2*b*(x-pi));
g2 = -a*exp(-b*((x-pi).^2 + (y-pi).^2 + (z-pi).^2)).*(-2*b*(y-pi));
for i = 0:11
    f =  f - c*cos(x)*cos(y)*exp(-beta*((x-r*sin(i/2)-pi)^2 + (y-r*cos(i/2)-pi)^2));
    g1 = g1 + c.*sin(x).*cos(y).*exp(-beta.*((x-r.*sin(i/2)-pi).^2 + (y-r.*cos(i/2)-pi).^2))+...
        c.*cos(x).*cos(y).*exp(-beta.*((x-r.*sin(i/2)-pi).^2 + (y-r.*cos(i/2)-pi).^2))*2*beta.*(x-r.*sin(i/2)-pi);
    g2 = g2 + c.*cos(x).*sin(y).*exp(-beta.*((x-r.*sin(i/2)-pi).^2 + (y-r.*cos(i/2)-pi).^2))+...
        c.*cos(x).*cos(y).*exp(-beta.*((x-r.*sin(i/2)-pi).^2 + (y-r.*cos(i/2)-pi).^2))*2*beta.*(y-r.*cos(i/2)-pi);
end
end
