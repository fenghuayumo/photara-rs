"""OMM alpha-card comparison: warm tracing cost and linear image error."""
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import photara_rs as pr
from PIL import Image


def render_case(path, disabled):
    os.environ["PHOTARA_DISABLE_OMM"] = "1" if disabled else "0"
    with pr.Scene(512, 512, samples=64, max_bounces=2, denoise=False,
                  adaptive_sampling=False) as scene:
        scene.world.color = (.2,.3,.5)
        scene.camera.location = (0,0,-4)
        scene.camera.target = (0,0,0)
        material = scene.materials.new("leaf", base_color=(.2,.7,.1), roughness=1)
        material.base_color_texture = {"path": str(path), "colorspace": "data"}
        material.configure_alpha_clip(.5)
        for layer in range(6):
            z = .15 * layer
            mesh = scene.meshes.new(str(layer), [(-2,-2,z),(2,-2,z),(2,2,z),(-2,2,z)],
                                     [(0,1,2),(0,2,3)], uvs=[(0,0),(1,0),(1,1),(0,1)],
                                     material=material)
            scene.objects.new(str(layer), mesh)
        begin = time.perf_counter()
        image = scene.render(seed=9).rgb.copy()
        cold = (time.perf_counter()-begin)*1000
        timings = []
        for _ in range(5):
            begin = time.perf_counter()
            image = scene.render(seed=9).rgb.copy()
            timings.append((time.perf_counter()-begin)*1000)
        return image, {"warm_render_capture_median_ms": float(np.median(timings)),
                        "cold_render_ms": cold, "stats": scene.omm_stats()}


def main():
    previous = os.environ.get("PHOTARA_DISABLE_OMM")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"leaf.png"
            rgba = np.full((256,256,4),255,dtype=np.uint8)
            rgba[:,:,3] = 0
            rgba[88:168,104:152,3] = 255
            Image.fromarray(rgba).save(path)
            off, off_stats = render_case(path, True)
            on, on_stats = render_case(path, False)
            metrics = {"off":off_stats,"on":on_stats,
                       "rmse":float(np.sqrt(np.mean((off-on)**2))),
                       "mean_off":float(off.mean()),"mean_on":float(on.mean()),
                       "scope":"512x512 64 spp; six alpha-tested cards; Reference; CPU AOV capture included"}
            out = Path("test_outputs/omm");out.mkdir(parents=True,exist_ok=True)
            (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
            print(json.dumps(metrics,indent=2))
    finally:
        if previous is None: os.environ.pop("PHOTARA_DISABLE_OMM",None)
        else: os.environ["PHOTARA_DISABLE_OMM"]=previous

if __name__ == "__main__": main()
