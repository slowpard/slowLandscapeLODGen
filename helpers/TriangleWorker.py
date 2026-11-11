#Runs Triangle in a separate process (NifSaver.triangulate_safe), so a crash or hang only takes down the child.
#Keep this module free of other imports: in the exe the multiprocessing child imports it on its own (the main script
#and its time.clock patch for pyffi never run there); from source it runs as a plain script instead.
import os
import sys
import numpy as np
import triangle


def _triangle_worker(in_path, out_path, flags, stderr_path=None):
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0002 | 0x0001)

    if stderr_path is not None:
        f = open(stderr_path, "w")
        os.dup2(f.fileno(), sys.stderr.fileno())
        os.dup2(f.fileno(), sys.stdout.fileno()) #Triangle prints its errors to stdout
    data = np.load(in_path)
    verts = data["vertices"].astype(np.float64).copy()
    segs = data["segments"].astype(np.int32).copy()
    result = triangle.triangulate({"vertices": verts, "segments": segs}, flags)
    np.savez(out_path, vertices=result["vertices"], triangles=result["triangles"])


if __name__ == '__main__':
    #from source: python TriangleWorker.py <input> <output> <flags>, stdout/stderr already go to the error file
    _triangle_worker(*sys.argv[1:4])
