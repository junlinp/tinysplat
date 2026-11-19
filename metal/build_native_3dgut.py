"""Build the independent native Metal 3DGUT library (no PyTorch headers)."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
out=root/'metal/build';out.mkdir(exist_ok=True)
source=(root/'metal/native_3dgut_kernels.metal').read_text().replace('// COMMON_SOURCE',(root/'legacy/tinysplat/cpp/native_3dgut_common.h').read_text())
(out/'native_3dgut_source.h').write_text('static const char*gut_source=R"GUTMSL('+source+')GUTMSL";\n')
subprocess.run(['clang++','-std=c++17','-O3','-fobjc-arc','-shared','-fPIC',str(root/'metal/native_3dgut.mm'),'-framework','Metal','-framework','Foundation','-o',str(out/'libtinysplat_3dgut.dylib')],check=True)
print(out/'libtinysplat_3dgut.dylib')
