% function [xams, xnos] = synthetic3()
% function [xnos, ynos] = synthetic3()
clear;
% generate a random point as the initial point
r = 2;  % 1.3, 1.5
i = randi(11); % which hole
delta = 0.; % 0.3
d = rand([1,2])*delta; % the distance to the hole
xy = d+ pi+[(r+delta)*sin(i/2),(r+delta)*cos(i/2)];
% xy = [2.9914,    4.2223];
% xy = [2.4699,    3.9774];
x = xy(1);
y = xy(2);

% x = rand(1)*4-2;
% y = rand(1)*4-2;
alpha = 5e-2;   %5e-1;  5e-2;  % 2e-1
[xams, yams] = deal(x,y);
[xnos, ynos] = deal(x,y);
[xiter, yiter] = deal(x,y);

[m1ams, m2ams] = deal(0,0);
[m1nos, m2nos] = deal(0,0);
[v1ams, v2ams] = deal(0,0);
[v1ams_max, v2ams_max] = deal(0,0);
[v1nos, v2nos] = deal(0,0);

[xadam, yadam] = deal(x,y);
[m1adam, m2adam] = deal(0,0);
[v1adam, v2adam] = deal(0,0);


max_iter = 500000;
beta1 = 0.9;
beta2 = 0.9;
gamma=0.1;
B = 0;
X1 = zeros(1,max_iter);
Y1 = zeros(1,max_iter);
X2 = zeros(1,max_iter);
Y2 = zeros(1,max_iter);
X3 = zeros(1,max_iter);
Y3 = zeros(1,max_iter);
for t=1:max_iter
    [~,g1,g2] = drill(xams,yams);
    X1(t) = xams;
    Y1(t) = yams;
    m1ams = beta1*m1ams + (1-beta1)*g1;
    m2ams = beta1*m2ams + (1-beta1)*g2;
    v1ams = beta2*v1ams + (1-beta2)*g1^2;
    v2ams = beta2*v2ams + (1-beta2)*g2^2;
    v1ams_max = max(v1ams_max, v1ams);
    v2ams_max = max(v2ams_max, v2ams);
    xams = xams - alpha*m1ams/sqrt(v1ams_max);
    yams= yams - alpha*m2ams/sqrt(v2ams_max);
    LR1(t) = alpha/sqrt(v1ams_max);
    LR2(t) = alpha/sqrt(v2ams_max);
    
end

for t=1:max_iter
    [~,g1,g2] = drill(xnos,ynos);
    X2(t) = xnos;
    Y2(t) = ynos;
    m1nos = beta1*m1nos + (1-beta1)*g1;
    m2nos = beta1*m2nos + (1-beta1)*g2;
    b = t^(-gamma);
    beta2_nos = B/(B+b);
    B = B + b;
    v1nos = beta2_nos*v1nos + (1-beta2_nos)*g1^2;
    v2nos = beta2_nos*v2nos + (1-beta2_nos)*g2^2;

    xnos = xnos - alpha*m1nos/sqrt(v1nos);
    ynos= ynos - alpha*m2nos/sqrt(v2nos);
    LR3(t) = alpha/sqrt(v1nos);
    LR4(t) = alpha/sqrt(v2nos);
end

% 
% for t=1:max_iter
%     [~,g1,g2] = drill(xadam,yadam);
%     X3(t) = xadam;
%     Y3(t) = yadam;
%     m1adam = beta1*m1adam + (1-beta1)*g1;
%     m2adam = beta1*m2adam + (1-beta1)*g2;
%     v1adam = beta2*v1adam + (1-beta2)*g1^2;
%     v2adam = beta2*v2adam + (1-beta2)*g2^2;
%     xadam = xadam - alpha*m1adam/sqrt(v1adam);
%     yadam = yadam - alpha*m2adam/sqrt(v2adam);
%     LR5(t) = alpha/sqrt(v1adam);
%     LR6(t) = alpha/sqrt(v2adam);
%     
% end

% g1_o = 0;
% g2_0 = 0;
% X4 = zeros(1,max_iter);
% Y4 = zeros(1,max_iter);
% for t=1:max_iter
%     if t == 1
%         deltax = 1;
%         deltay = 1;
%         deltag1 = 1;
%         deltag2 = 1;
%         [~,g1,g2] = drill(xiter,yiter);
%         g1_new = g1;
%         g2_new = g2;
%         x_new = xiter;
%         y_new = yiter;
%         
%     else
%         [~,g1,g2] = drill(xiter,yiter);
%         g1_old = g1_new;
%         g2_old = g2_new;
%         g1_new = g1;
%         g2_new = g2;
%         
%         deltax = x_old - x_new;
%         deltay = y_old - y_new;
%         deltag1 = g1_new - g1_old;
%         deltag2 = g2_new - g2_old;
%     end
%     X4(t) = xiter;
%     Y4(t) = yiter;
%     xiter = xiter - alpha*g1*deltax/deltag1 ;
%     yiter= yiter - alpha*g2*deltay/deltag2;
%     x_old = x_new;
%     y_old = y_new;
%     x_new = xiter;
%     y_new = yiter;
%     
% end