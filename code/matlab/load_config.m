function cfg = load_config()
cfg = jsondecode(fileread(fullfile(hop3_root(), 'config.json')));
end
