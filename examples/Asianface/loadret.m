function ret = loadret
ret = zeros(50, 10);
for i = 0:9
    fid = fopen(sprintf('log%d', i));
    for j = 1:50
        while 1
            line = fgets(fid);
            if strfind(line, 'Test net output #0:')
                ret(j, i + 1) = str2double(line(82:end));
                break;
            end
        end
    end
    fclose(fid);
end