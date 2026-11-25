-- RPPS: what each decoded frame was, as fields to filter on, before its own link type's dissector
--
-- Captures written with RPPS's header (rp.source.PcapSink(target, rpps=True); `rpps rx ... --pcap - --pcap-rpps`) are
-- on LINKTYPE_USER0: "RPPS", a version byte, the header's length (16 bits), TLVs (type byte, 16-bit length, value),
-- then the packet. This shows the TLVs and hands the packet to the dissector of its pcap link type (the ones RPPS's
-- config/linktype has: a new one there is a line in ENCAP below).
--
--   wireshark -X lua_script:tools/wireshark/rpps.lua -k -i -      (or copy it to Wireshark's personal plugins folder)
--
-- Filters: rpps.crc_ok == 0 (failed checks), rpps.profile == "ble/adv", rpps.kind == "pdu", rpps.check,
-- rpps.freq, rpps.quality, rpps.field == "type.name=ADV_IND" (each field the capture rule names, as name=value).

local rpps = Proto("rpps", "RPPS decoded frame")
local f = rpps.fields
f.version = ProtoField.uint8("rpps.version", "Version")
f.profile = ProtoField.string("rpps.profile", "Profile")
f.kind = ProtoField.string("rpps.kind", "Message kind")
f.check = ProtoField.string("rpps.check", "Check")
f.crc_ok = ProtoField.uint8("rpps.crc_ok", "Check passed", base.DEC, { [0] = "failed", [1] = "passed", [2] = "not checked" })
f.sample = ProtoField.int64("rpps.sample", "Sample")
f.time = ProtoField.double("rpps.time", "Time [s]")
f.freq = ProtoField.double("rpps.freq", "Frequency [Hz]")
f.quality = ProtoField.double("rpps.quality", "Quality [dB]")
f.dlt = ProtoField.uint16("rpps.dlt", "Link type (pcap)")
f.dissector = ProtoField.string("rpps.dissector", "Dissector")
f.fields = ProtoField.string("rpps.fields", "Fields")
f.field = ProtoField.string("rpps.field", "Field")

local failed = ProtoExpert.new("rpps.failed", "The frame's check failed", expert.group.CHECKSUM, expert.severity.WARN)
rpps.experts = { failed }

local encaps = DissectorTable.get("wtap_encap")
local W = wtap_encaps or wtap
-- pcap link types (config/linktype) to Wireshark's encapsulations, which its wtap_encap table is keyed by
local ENCAP = {
    [1] = W.ETHERNET, [3] = W.AX25, [9] = W.PPP, [101] = W.RAW_IP, [104] = W.CHDLC, [105] = W.IEEE_802_11,
    [127] = W.IEEE_802_11_RADIOTAP, [195] = W.IEEE802_15_4, [202] = W.AX25_KISS, [228] = W.RAW_IP4, [229] = W.RAW_IP6,
    [230] = W.IEEE802_15_4_NOFCS, [243] = W.MPEG_2_TS, [251] = W.BLUETOOTH_LE_LL, [252] = W.WIRESHARK_UPPER_PDU,
    [256] = W.BLUETOOTH_LE_LL_WITH_PHDR,
}

function rpps.dissector(tvb, pinfo, tree)
    if tvb:len() < 7 or tvb(0, 4):string() ~= "RPPS" then
        return 0
    end
    local hlen = tvb(5, 2):uint()
    local t = tree:add(rpps, tvb(0, hlen))
    t:add(f.version, tvb(4, 1))
    local i, dlt, ok, profile, kind, name = 7, nil, nil, "", "", nil
    while i + 3 <= hlen do
        local typ, n = tvb(i, 1):uint(), tvb(i + 1, 2):uint()
        local v = tvb(i + 3, n)
        if typ == 1 then
            profile = v:string()
            t:add(f.profile, v)
        elseif typ == 2 then
            kind = v:string()
            t:add(f.kind, v)
        elseif typ == 3 then
            t:add(f.check, v)
        elseif typ == 4 then
            ok = v:uint()
            local it = t:add(f.crc_ok, v)
            if ok == 0 then
                it:add_proto_expert_info(failed)
            end
        elseif typ == 5 then
            t:add(f.sample, v, v:int64())
        elseif typ == 6 then
            t:add(f.time, v)
        elseif typ == 7 then
            t:add(f.freq, v)
        elseif typ == 8 then
            t:add(f.quality, v)
        elseif typ == 9 then
            dlt = v:uint()
            t:add(f.dlt, v)
        elseif typ == 10 then
            name = v:string()
            t:add(f.dissector, v)
        elseif typ == 11 then
            local s = v:string()
            local ft = t:add(f.fields, v)
            for k, val in string.gmatch(s, '"([^"]+)":%s*"([^"]*)"') do
                ft:add(f.field, v, k .. "=" .. val)
            end
        end
        i = i + 3 + n
    end
    if hlen < tvb:len() then
        local payload = tvb(hlen):tvb()
        local found, d = false, nil
        if name ~= nil then
            found, d = pcall(Dissector.get, name) -- by name first: some encapsulations dispatch on the frame's own (USER0)
        end
        if found and d ~= nil then
            d:call(payload, pinfo, tree)
        elseif dlt ~= nil and ENCAP[dlt] ~= nil then
            encaps:try(ENCAP[dlt], payload, pinfo, tree)
        else
            Dissector.get("data"):call(payload, pinfo, tree)
        end
    end
    local mark = (ok == 0) and " [check FAILED]" or ""
    pinfo.cols.info:prepend(profile .. " " .. kind .. mark .. ": ")
    return tvb:len()
end

encaps:add(W.USER0, rpps)
