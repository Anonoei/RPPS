"""LTE broadcast RRC messages (3GPP TS 36.331): build an ASN.1 module from the spec and decode BCCH-DL-SCH messages (SIB1, SI messages)

    python projects/lte/lte_rrc.py build 36331-j30.docx eutra.asn      # once: extract the ASN.1 from the spec's .docx
    python projects/lte/lte_rrc.py decode eutra.asn 484c4690...         # hex of a decoded transport block

Needs `pip install asn1tools` (not a dependency of RPPS; use a venv). The spec's ASN.1 is the single module EUTRA-RRC-Definitions between the
"-- ASN1START" and "-- ASN1STOP" markers. Two repairs make it compile with asn1tools: the document extraction must keep the tabs between field names and
types (a bare regex over word/document.xml drops them), and the parameterized type `SetupRelease {X}` is expanded textually to `CHOICE {release NULL, setup X}`.
Download: https://www.3gpp.org/ftp/Specs/archive/36_series/36.331/36331-j30.zip (Rel-19; any release works for the broadcast messages).
"""
import html
import re
import sys
import zipfile

def build(docx, out):
    x = zipfile.ZipFile(docx).read("word/document.xml").decode()
    x = x.replace("<w:tab/>", " ")
    x = re.sub(r"</w:p>", "\n", x); x = re.sub(r"<w:br/>", "\n", x)
    t = html.unescape(re.sub(r"<[^>]+>", "", x))
    blocks = re.findall(r"-- ASN1START(.*?)-- ASN1STOP", t, re.S)
    asn = "\n".join(blocks)
    end = asn.index("\nEND") # the first module is EUTRA-RRC-Definitions
    asn = asn[asn.index("EUTRA-RRC-Definitions DEFINITIONS"):end + 4]
    asn = re.sub(r"SetupRelease\s*\{\s*ElementTypeParam\s*\}\s*::=\s*CHOICE\s*\{\s*release\s+NULL,\s*setup\s+ElementTypeParam\s*\}", "", asn)
    asn = re.sub(r"SetupRelease\s*\{\s*([A-Za-z0-9\-]+)\s*\}", r"CHOICE {release NULL, setup \1}", asn)
    open(out, "w").write(asn)
    return out

def load(asn_path):
    import asn1tools
    return asn1tools.compile_files(asn_path, "uper")

def decode_bcch_dl_sch(codec, data):
    """BCCH-DL-SCH-Message from the bytes of a transport block (trailing zero padding is fine)"""
    return codec.decode("BCCH-DL-SCH-Message", bytes(data))

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "build":
        print(build(sys.argv[2], sys.argv[3]))
    elif len(sys.argv) >= 4 and sys.argv[1] == "decode":
        import pprint
        pprint.pprint(decode_bcch_dl_sch(load(sys.argv[2]), bytes.fromhex(sys.argv[3])), width=140)
    else:
        print(__doc__)
