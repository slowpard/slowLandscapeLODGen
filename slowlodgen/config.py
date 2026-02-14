import dataclasses
from dataclasses import dataclass, field
import logging
from typing import List, Optional, Literal, Union
import tomllib
import os

@dataclass
class GeneralSettings:
    generate_meshes: bool = True
    generate_normal_maps: bool = True
    generate_color_maps: bool = True
    use_enginebugfixes_naming_scheme: Union[bool, Literal["Auto"]] = "Auto"

@dataclass
class MeshSettings:
    vertices_batch: float = 0.8
    mesh_min_error: int = 500_000
    current_vert_abs_minimum: int = 1089
    first_loop_min_error: int = 500
    target_triangles: int = 62_000
    multithreaded: bool = True
    threads: int = 8

    @property
    def target_vertices(self) -> int:
        return int(self.target_triangles / 2)




@dataclass
class TextureSettings:
    texture_dimension: int = 2048
    internal_dimension: int = 2048
    dds_encode_quality: int = 10

    gauss_blending_falloff: float = 2.5


    def __post_init__(self):
        for name in ("texture_dimension", "internal_dimension"):
            val = getattr(self, name)
            if val not in {512, 1024, 2048, 4096, 8192, 16384}:
                logging.warning(f"'{name}={val}' must be a power of 2 from 512 to 16384 � defaulting to 2048.")
                setattr(self, name, 2048)
        if self.internal_dimension < self.texture_dimension:
            self.internal_dimension = self.texture_dimension

    @property
    def dimension_mult(self) -> float:
        return self.internal_dimension / self.texture_dimension

    @property
    def sample_res(self) -> int:
        return int(self.internal_dimension / 32 / 16)

    @property
    def chunk_dim(self) -> int:
        return int(round(self.sample_res / 2))


@dataclass
class NormalMapSettings:
    texture_dimension: int = 4096
    boost: float = 1.0
    z_boost: float = 1.0

    shadows_enabled: bool = False
    shadow_strength: float = 1.0
    compression: bool = True

    @property
    def internal_dimension(self):
        return max(self.texture_dimension, 1024)

    def __post_init__(self):
        for name in ("texture_dimension",):
            val = getattr(self, name)
            if val not in {256, 512, 1024, 2048, 4096, 8192, 16384}:
                logging.warning(f"'{name}={val}' must be a power of 2 from 512 to 16384 � defaulting to 2048.")
                setattr(self, name, 2048)




@dataclass
class NormalMapNoiseSettings:
    noise_enabled: bool = False
    noise_scale: int = 20
    noise_intensity: int = 8
    noise_octaves: int = 15
    noise_persistence: float = 0.5
    noise_lacunarity: float = 2.0



@dataclass
class AOSettings:
    enabled: bool = True
    seek_range: int = 512
    strength: float = 1.0
    gamma_min: float = 0.0
    gamma_exp: float = 1.75
    gamma_max: float = 0.85


@dataclass
class PathSettings:
    oblivion_folder: Optional[str] = None
    plugins_txt: Optional[str] = None




@dataclass
class Config:
    debug_level: str = "INFO"
    general: GeneralSettings = field(default_factory=GeneralSettings)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    texture: TextureSettings = field(default_factory=TextureSettings)
    normal_map: NormalMapSettings = field(default_factory=NormalMapSettings)
    normal_map_noise: NormalMapNoiseSettings = field(default_factory=NormalMapNoiseSettings)
    ao: AOSettings = field(default_factory=AOSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    hardcoded_bsas: List[str] = field(default_factory=lambda: [
        "Oblivion - Meshes.bsa", "Oblivion - Misc.bsa",
        "Oblivion - Textures - Compressed.bsa",
        "N - Meshes.bsa", "N - Textures1.bsa", "N - Textures2.bsa", "N - Misc.bsa",
    ])
    force_enabled_plugins: List[str] = field(default_factory=list)
    excluded_worldspaces: List[str] = field(default_factory=list)


    def apply_overrides(self):
        if not self.mesh.multithreaded:
            self.mesh.threads = 1

        self.mesh.target_triangles = max(0, min(62000, self.mesh.target_triangles))


def load_config(path):
    cfg = Config()

    if not os.path.exists(path):
        logging.critical(f"{path} not found - using defaults.")
        return cfg

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    general_raw = raw.get("general", {})
    cfg.debug_level = general_raw.pop("debug_level", cfg.debug_level)
    cfg.force_enabled_plugins = general_raw.pop("force_enabled_plugins", [])
    cfg.excluded_worldspaces = general_raw.pop("excluded_worldspaces", [])

    config_sections = {
        "general": cfg.general, "mesh": cfg.mesh,
        "texture": cfg.texture, "normal_map": cfg.normal_map,
        "normal_map_noise": cfg.normal_map_noise,
        "ao": cfg.ao, "paths": cfg.paths,
    }

    for name, dc in config_sections.items():
        if name not in raw:      
            continue
        known = {f.name: f.type for f in dataclasses.fields(dc)}
        for key, val in raw[name].items():
            if key not in known:
                logging.warning(f"Unknown config key '{name}.{key}' - skipping.")
                continue
            expected = known[key]
            if expected is float and isinstance(val, int):
                val = float(val)
            try:
                if not isinstance(val, eval(expected) if isinstance(expected, str) else expected):
                    logging.warning(f"'{key}': expected {expected}, got {type(val).__name__} � using default.")
                    continue
            except TypeError:
                #complex type like Union[bool, Literal["Auto"]] � just accept the value
                pass
            if isinstance(val, str):
                val = val.strip()
            setattr(dc, key, val)

    if "bsa" in raw and "hardcoded_bsas" in raw["bsa"]:
        cfg.hardcoded_bsas = [s.strip() for s in raw["bsa"]["hardcoded_bsas"]]

    cfg.apply_overrides()
    return cfg
