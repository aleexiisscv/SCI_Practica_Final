clc; clear;

%% === ENTRENAMIENTO FFNN ROBUSTA (sin obstáculos) - SOLO SESGO FIJO ===
load('train_fuzzy_sin_obs_ALL.mat','training_data');

%% 1) Selección de sensores y salidas
sensor_cols = 1:6;           % sonar_0..sonar_5 (6 frontales)
vel_col     = 13;            % vel_kmh
steer_col   = 14;            % steering_deg

X = training_data(:, sensor_cols)';                 % 6 x N
Y = training_data(:, [vel_col, steer_col])';        % 2 x N -> [vel; steer]

%% 2) Limpieza + clipping físico
X(isnan(X) | isinf(X)) = 5.0;
X = max(0.1, min(15.0, X));                         % rango sonar

Y(isnan(Y) | isinf(Y)) = 0.0;
Y(1,:) = max(0.0,  min(50.0,  Y(1,:)));             % velocidad [0,50] km/h
Y(2,:) = max(-90.0, min(90.0, Y(2,:)));             % steering [-90,90] deg

X = double(X);
Y = double(Y);

%% 3) Steering “seguro”: suavizado + SESGO FIJO
Y(2,:) = movmean(Y(2,:), 5);

steer_bias = -2.43;              % AJUSTA AQUÍ (micro pasos: -2.38, -2.40, -2.45)
Y(2,:) = Y(2,:) + steer_bias;

Y(2,:) = max(-60.0, min(60.0, Y(2,:)));            % límite final aprendido

%% 4) Barajar
N = size(X,2);
rng(7);
idx = randperm(N);
X = X(:,idx);
Y = Y(:,idx);

%% 5) Data augmentation (robustez): ruido crudo + dropout de sensores
X_aug = X + 0.3*randn(size(X));                     % si te hace inestable: baja a 0.2
X_aug = max(0.1, min(15.0, X_aug));

X_drop = X;
p = 0.02;
mask = rand(size(X_drop)) < p;
X_drop(mask) = 15.0;

Xall = [X, X_aug, X_drop];
Yall = [Y, Y,     Y];

%% 6) Red feedforward con regularización
net = feedforwardnet([25 15], 'trainbr');
net.layers{end}.transferFcn = 'tansig';

% Normalización interna (Simulink -> salidas reales)
net.inputs{1}.processFcns  = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};

net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

net.trainParam.epochs = 500;

%% 7) Entrenar
[net, tr] = train(net, Xall, Yall);

%% 8) Evaluación rápida en TEST
testInd = tr.testInd;
Xt = Xall(:, testInd);
Yt = Yall(:, testInd);

Yp = net(Xt);
err = Yp - Yt;

rmse_vel   = sqrt(mean(err(1,:).^2));
rmse_steer = sqrt(mean(err(2,:).^2));

fprintf('\n=== RMSE en TEST (unidades reales) ===\n');
fprintf('Velocidad (km/h): %.3f\n', rmse_vel);
fprintf('Steering (deg):   %.3f\n', rmse_steer);

fprintf('\nRangos predichos en TEST:\n');
fprintf('Vel pred:   [%.2f, %.2f] km/h\n', min(Yp(1,:)), max(Yp(1,:)));
fprintf('Steer pred: [%.2f, %.2f] deg\n',  min(Yp(2,:)), max(Yp(2,:)));

%% 9) Guardar y generar bloque Simulink
save('net_ff_6front_solo_sesgo.mat', ...
     'net','tr','sensor_cols','vel_col','steer_col');

gensim(net, 0.1);
