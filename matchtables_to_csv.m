% Directory containing the .mat files
matRootDir = "\\znas\Lab\Share\UNITMATCHTABLES_ENNY_CELIAN_JULIE\FullAnimal_KSChanMap";
csvRootDir = 'C:\Users\suyas\UNSEEN_DATA';


% mice = ["AL031", "AL032", "AL036", "AV008", "CB015", "CB016", "CB017", "CB018", "CB020", "EB019"];
mice = ["AV015", "AV021", "AV049", "EB014", "FT033", "FT039", "JF084"];
sizes = [];

for mouse = mice
    mouse_path = matRootDir + "\" + mouse;
    probes = struct2cell(dir(mouse_path));
    probes = probes(1,3:end);
    for probe = probes
        if probe == "." || probe == ".."
            continue
        end
        probe_path = mouse_path + "\" + probe;
        locations = struct2cell(dir(probe_path));
        locations = locations(1,3:end);
        for loc = locations
            if probe == "." || probe == ".."
                continue
            end
            loc_path = probe_path + "\" + loc;
            path = loc_path + "\UnitMatch\UnitMatch.mat" ;
            UM = load(path);
            mt = UM.MatchTable;
            sizes(end+1)=numel(mt);
            writetable(mt, csvRootDir + "\" + mouse + "\" + probe + "\" + loc + "\matchtable.csv")
        end
    end
end