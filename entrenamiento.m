clear; clc; close all;

% ---- 1) Cargar y concatenar vueltas ----
files = dir('training_data_v*.mat');
assert(~isempty(files), 'No se encontraron training_data_v*.mat en el directorio.');

inputs_all  = [];
targets_all = [];

for k = 1:length(files)
    S = load(files(k).name);

    % S.inputs:  4 x N
    % S.targets: 2 x N
    if ~isfield(S,'inputs') || ~isfield(S,'targets')
        error('El archivo %s no tiene variables inputs/targets.', files(k).name);
    end

    inputs_all  = [inputs_all  S.inputs];
    targets_all = [targets_all S.targets];
end

fprintf('Total muestras: %d\n', size(inputs_all,2));
fprintf('Entradas: %d (esperado 4)\n', size(inputs_all,1));
fprintf('Salidas:  %d (esperado 2)\n', size(targets_all,1));

assert(size(inputs_all,1)==4, 'inputs_all debe ser 4xN (4 sensores).');
assert(size(targets_all,1)==2, 'targets_all debe ser 2xN ([vel; steering]).');

% ---- 2) Limpieza ligera por seguridad ----
inputs_all(isnan(inputs_all) | isinf(inputs_all)) = 5.0;
inputs_all = min(max(inputs_all,0),15);

targets_all(isnan(targets_all) | isinf(targets_all)) = 0;

% ---- 3) Crear y entrenar red (normalización interna) ----
net = feedforwardnet([20 10], 'trainlm');

% Clave: la red se normaliza sola (en Simulink no haces nada)
net.inputs{1}.processFcns  = {'mapminmax'};
net.outputs{1}.processFcns = {'mapminmax'};

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

net.trainParam.epochs   = 500;
net.trainParam.max_fail = 10;

[net, tr] = train(net, inputs_all, targets_all);

% ---- 4) Evaluación ----
outputs = net(inputs_all);              % ya en unidades reales
err = outputs - targets_all;

rmse_vel   = sqrt(mean(err(1,:).^2));
rmse_steer = sqrt(mean(err(2,:).^2));

fprintf('RMSE velocidad = %.3f km/h\n', rmse_vel);
fprintf('RMSE steering  = %.3f grados\n', rmse_steer);

% (opcional) Nerviosismo de steering por muestra (0.1s)
dsteer = diff(outputs(2,:));
fprintf('Pico |Δsteer| = %.3f grados/0.1s\n', max(abs(dsteer)));

% ---- 5) Generar bloque Simulink ----
Ts = 0.1;
gensim(net, Ts);
