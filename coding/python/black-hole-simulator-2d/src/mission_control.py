# src/mission_control.py
# Main mission controller with runtime mission selection

import moderngl_window as mglw
import moderngl

from missions.mission1_grid_blackhole import Mission1GridBlackHole
from missions.mission2_single_beam import Mission2SingleBeam
from missions.mission3_multiple_beams_no_collision import Mission3MultipleBeamsNoCollision
from missions.mission4_multiple_beams import Mission4MultipleBeams
from missions.mission5_units_schwarzschild import Mission5UnitsSchwarzschild
from missions.mission6_fixed_timestep import Mission6FixedTimestep
from missions.mission7_light_bending import Mission7LightBending
from missions.mission8_validation import Mission8Validation
from missions.mission9_redshift import Mission9Redshift

# ---------- Simple key binding manager (no external deps) ----------
class KeyBinder:
    """
    Centralized hotkey manager.
    - Register actions with one or more key-name variants and a callback.
    - Edge-triggered (fires only on press).
    - Groups let us enable/clear per-mission bindings cleanly.
    - Now also renders human-readable key names in help.
    """
    def __init__(self, wnd):
        self.wnd = wnd
        self.keys = wnd.keys
        self._held = set()
        # Each binding: {"label": str, "keys": [key_consts], "cb": func, "group": str, "key_labels": [str]}
        self._bindings = []
        self._alias_cache = {}  # cache: key name -> key const

    # --- Key lookup helpers -------------------------------------------------
    def key_const(self, *names):
        """Return the first existing key constant among provided backend name variants."""
        for name in names:
            if name in self._alias_cache:
                k = self._alias_cache[name]
                if k is not None:
                    return k
                continue
            k = getattr(self.keys, name, None)
            self._alias_cache[name] = k
            if k is not None:
                return k
        return None

    def _pretty_key_name(self, raw_name: str) -> str:
        """Make backend key names friendlier for humans."""
        if raw_name is None:
            return "?"
        # Canonicalize common names
        mapping = {
            "LEFT_BRACKET": "[",
            "RIGHT_BRACKET": "]",
            "EQUAL": "=",
            "MINUS": "-",
            "COMMA": ",",
            "PERIOD": ".",
            "APOSTROPHE": "'",
            "SEMICOLON": ";",
            "SPACE": "SPACE",
            "ESCAPE": "ESC",
            "PAGEDOWN": "PgDn",
            "PAGE_DOWN": "PgDn",
            "PAGEUP": "PgUp",
            "PAGE_UP": "PgUp",
            "KP_ADD": "Num+",
            "KP_SUBTRACT": "Num-",
            "NUM_0": "0",
            "NUM_1": "1",
            "NUM_2": "2",
            "NUM_3": "3",
            "NUM_4": "4",
            "NUM_5": "5",
            "NUM_6": "6",
            "NUM_7": "7",
            "NUM_8": "8",
            "NUM_9": "9",
            "_0": "0",
            "_1": "1",
            "_2": "2",
            "_3": "3",
            "_4": "4",
            "_5": "5",
            "_6": "6",
            "_7": "7",
            "_8": "8",
            "_9": "9",
            "OEM_COMMA": ",",
            "OEM_PERIOD": ".",
            "OEM_1": ";",
            "OEM_7": "'",
        }

        # Keep letters and arrows readable
        simple = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
        if raw_name in mapping:
            return mapping[raw_name]
        if raw_name.upper() in simple:
            return simple[raw_name.upper()]
        # fallback: return as-is (KEY_J -> KEY_J, but we try to shorten KEY_X -> X)
        if raw_name.startswith("KEY_") and len(raw_name) == 5:
            return raw_name[-1]
        return raw_name

    # --- API to add bindings -----------------------------------------------
    def add(self, label, callback, *key_name_variants, group="Global"):
        """
        Register a binding.
        label   : human-friendly label for help output.
        callback: function() called on press.
        key_name_variants:
            each item is either a string "J" or a tuple ("J","KEY_J") etc.
            Every item represents one physical key with multiple backend name fallbacks.
            Example:
              ("COMMA","OEM_COMMA")  -> means this action fires when COMMA is pressed,
                                        but if that name doesn't exist, use OEM_COMMA.
        group   : logical group (Global, Mission 2, Mission 6, ...).

        We also store 'key_labels' (pretty names) for help.
        """
        key_consts = []
        key_labels = []
        for variant in key_name_variants:
            if isinstance(variant, str):
                variant = (variant,)
            # resolve constant and remember the pretty display text
            chosen_name = None
            chosen_const = None
            for name in variant:
                k = self.key_const(name)
                if k is not None:
                    chosen_name = name
                    chosen_const = k
                    break
            key_consts.append(chosen_const)
            key_labels.append(self._pretty_key_name(chosen_name or variant[0]))
        self._bindings.append({
            "label": label,
            "keys": key_consts,
            "key_labels": key_labels,
            "cb": callback,
            "group": group
        })

    def clear_group(self, group):
        """Remove all bindings of a given group."""
        self._bindings = [b for b in self._bindings if b["group"] != group]

    # --- Polling / firing ---------------------------------------------------
    def _is_pressed(self, k):
        return bool(k) and (self.wnd.is_key_pressed(k) if hasattr(self.wnd, "is_key_pressed") else False)

    def poll(self):
        """Edge-triggered scan: when any key of a binding goes down, fire once."""
        for b in self._bindings:
            name = b["label"]
            is_any_down = any(self._is_pressed(k) for k in b["keys"] if k is not None)
            if is_any_down:
                if name not in self._held:
                    self._held.add(name)
                    try:
                        b["cb"]()
                    except Exception as e:
                        print(f"[KeyBinder] Error in '{name}': {e}")
            else:
                self._held.discard(name)

    # --- Help rendering -----------------------------------------------------
    def help_text(self):
        """
        Return grouped help text of active bindings with readable key hints.
        Example line: 'Pause (toggle) — SPACE'
        If multiple keys trigger the same binding, we join them with ' / '.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for b in self._bindings:
            # Filter out bindings with no keys resolved (just in case)
            keys_for_help = [k for k in b["key_labels"] if k not in (None, "?")]
            key_hint = " / ".join(keys_for_help) if keys_for_help else ""
            groups[b["group"]].append((b["label"], key_hint))

        lines = ["", "="*60, "Active Controls", "="*60]
        for g in sorted(groups.keys()):
            lines.append(f"[{g}]")
            for label, hint in groups[g]:
                if hint:
                    lines.append(f"  - {label} — {hint}")
                else:
                    lines.append(f"  - {label}")
            lines.append("")
        lines.append("="*60)
        return "\n".join(lines)


class MissionControl(mglw.WindowConfig):
    """Mission Control for Black Hole Simulation Demo"""
    gl_version = (3, 3)
    title = "SpaceD - Mission Control"
    window_size = (1600, 1000)
    aspect_ratio = 1100 / 800
    resizable = False
    # Store mission number as a class attribute for runtime selection
    selected_mission_number = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.BLEND)
        self.ctx._window = self.wnd
        self.w, self.h = self.window_size

        # 1) Сначала создаём KeyBinder (он понадобится в _initialize_mission)
        self.kb = KeyBinder(self.wnd)

        # 2) Регистрируем глобальные биндинги (они не зависят от активной миссии)
        self.kb.add("Quit",               lambda: self.wnd.close(), ("ESCAPE",), ("Q",), group="Global")
        self.kb.add("Pause (toggle)",     lambda: setattr(self.active_mission, "paused",
                                                        not getattr(self.active_mission, "paused", False)),
                    ("SPACE",), group="Global")

        self.kb.add("BH radius +",        lambda: setattr(self.active_mission, "rs_px",
                                                        getattr(self.active_mission, "rs_px", 80.0) * 1.1),
                    ("EQUAL",), ("KP_ADD",), group="Global")
        self.kb.add("BH radius -",        lambda: setattr(self.active_mission, "rs_px",
                                                        getattr(self.active_mission, "rs_px", 80.0) / 1.1),
                    ("MINUS",), ("KP_SUBTRACT",), group="Global")

        self.kb.add("Grid spacing -",     lambda: setattr(self.active_mission, "grid_gap_px",
                                                        max(4.0, float(getattr(self.active_mission,"grid_gap_px",32.0))/1.2)),
                    ("LEFT_BRACKET",), group="Global")
        self.kb.add("Grid spacing +",     lambda: setattr(self.active_mission, "grid_gap_px",
                                                        min(256.0, float(getattr(self.active_mission,"grid_gap_px",32.0))*1.2)),
                    ("RIGHT_BRACKET",), group="Global")

        self.kb.add("Show help in console", lambda: print(self.kb.help_text()), ("H",), group="Global")

        # 3) Теперь можно создать миссию (kb уже существует и готов)
        self.active_mission = None
        self._initialize_mission(MissionControl.selected_mission_number)

        print(f"Running: {self.active_mission.get_name() if self.active_mission else 'No mission selected'}")

    def _initialize_mission(self, mission_number):
        """Initialize the selected mission by number"""
        if mission_number == 1:
            self.active_mission = Mission1GridBlackHole(self.ctx, self.w, self.h)
        elif mission_number == 2:
            self.active_mission = Mission2SingleBeam(self.ctx, self.w, self.h)
        elif mission_number == 3:
            self.active_mission = Mission3MultipleBeamsNoCollision(self.ctx, self.w, self.h)
        elif mission_number == 4:
            self.active_mission = Mission4MultipleBeams(self.ctx, self.w, self.h)
        elif mission_number == 5:
            self.active_mission = Mission5UnitsSchwarzschild(self.ctx, self.w, self.h)
        elif mission_number == 6:
            self.active_mission = Mission6FixedTimestep(self.ctx, self.w, self.h)
        elif mission_number == 7:
            self.active_mission = Mission7LightBending(self.ctx, self.w, self.h)
        elif mission_number == 8:
            self.active_mission = Mission8Validation(self.ctx, self.w, self.h)
        elif mission_number == 9:
            self.active_mission = Mission9Redshift(self.ctx, self.w, self.h)
        else:
            print("ERROR: Invalid mission number! (1-9)")
            return

        self.active_mission.initialize()

        # Убираем старые биндинги миссий
        for g in ("Mission 2", "Mission 3", "Mission 4", "Mission 5", "Mission 6", "Mission 7", "Mission 8", "Mission 9"):
            self.kb.clear_group(g)

        # Регистрируем биндинги активной миссии
        self._register_mission_bindings()


    def _register_mission_bindings(self):
        """
        Register mission-specific hotkeys in the central KeyBinder.
        We avoid conflicts: e.g. in Mission 6 we do NOT bind Mission 3 spacing keys.
        """
        m = self.active_mission

        # ----- Mission 2: single beam -----
        if isinstance(m, Mission2SingleBeam):
            self.kb.add("Beam speed slower (M2)", lambda: setattr(m, "beam_speed_px",
                                                                  max(10.0, float(getattr(m,"beam_speed_px",200.0))/1.2)),
                        ("_1","NUM_1","NUMBER_1","ONE","N1","K1"), group="Mission 2")
            self.kb.add("Beam speed faster (M2)", lambda: setattr(m, "beam_speed_px",
                                                                  min(2000.0, float(getattr(m,"beam_speed_px",200.0))*1.2)),
                        ("_2","NUM_2","NUMBER_2","TWO","N2","K2"), group="Mission 2")
            self.kb.add("Particle size - (M2)",   lambda: setattr(m, "point_size_px",
                                                                  max(2.0, float(getattr(m,"point_size_px",6.0))/1.2)),
                        ("_9","NUM_9","NUMBER_9","NINE","N9","K9"), group="Mission 2")
            self.kb.add("Particle size + (M2)",   lambda: setattr(m, "point_size_px",
                                                                  min(64.0, float(getattr(m,"point_size_px",6.0))*1.2)),
                        ("_0","NUM_0","NUMBER_0","ZERO","N0","K0"), group="Mission 2")
            # Optional: color cycle & reset if helpers exist
            if hasattr(m, "_cycle_color"):
                self.kb.add("Beam color cycle (M2)",  lambda: m._cycle_color(), ("C",), group="Mission 2")
            if hasattr(m, "_reset_beam"):
                self.kb.add("Reset beam (M2)",        lambda: m._reset_beam(),  ("R",), group="Mission 2")

        # ----- Mission 3: multiple beams (spacing, reset) -----
        # NOTE: Skip spacing keys if Mission 6 is active to avoid conflict with its UI.
        if isinstance(m, Mission3MultipleBeamsNoCollision) and not isinstance(m, Mission6FixedTimestep):
            if hasattr(m, "reset_all_beams"):
                self.kb.add("Reset all beams (M3)", lambda: m.reset_all_beams(), ("R",), group="Mission 3")
            if hasattr(m, "_repack_y_positions"):
                self.kb.add("Beam spacing - (M3)",  lambda: (setattr(m, "beam_spacing_px",
                                                                     max(4.0, float(m.beam_spacing_px)/1.15)),
                                                             m._repack_y_positions()),
                            ("COMMA","OEM_COMMA"), group="Mission 3")
                self.kb.add("Beam spacing + (M3)",  lambda: (setattr(m, "beam_spacing_px",
                                                                     min(80.0, float(m.beam_spacing_px)*1.15)),
                                                             m._repack_y_positions()),
                            ("PERIOD","OEM_PERIOD"), group="Mission 3")

        # ----- Mission 4: collisions -----
        if isinstance(m, Mission4MultipleBeams):
            if hasattr(m, "toggle_respawn"):
                self.kb.add("Toggle respawn (M4)", lambda: m.toggle_respawn(), ("T",), group="Mission 4")
            if hasattr(m, "clear_hits"):
                self.kb.add("Clear hits (M4)",     lambda: m.clear_hits(),    ("Y",), group="Mission 4")

        # ----- Mission 5: SI units -----
        if isinstance(m, Mission5UnitsSchwarzschild):
            self.kb.add("Zoom in (M5)",     lambda: m.zoom_in(),  ("Z",), group="Mission 5")
            self.kb.add("Zoom out (M5)",    lambda: m.zoom_out(), ("X",), group="Mission 5")
            self.kb.add("Mass up (M5)",     lambda: m.mass_up(),  ("M",), group="Mission 5")
            self.kb.add("Mass down (M5)",   lambda: m.mass_down(),("N",), group="Mission 5")
            self.kb.add("Slower time (M5)", lambda: m.slower(),   ("_1","NUM_1","NUMBER_1","ONE"), group="Mission 5")
            self.kb.add("Faster time (M5)", lambda: m.faster(),   ("_2","NUM_2","NUMBER_2","TWO"), group="Mission 5")
            self.kb.add("Grid lock toggle (M5)", lambda: m.toggle_grid_lock(), ("V",), group="Mission 5")
            self.kb.add("Print SI state (M5)",   lambda: m._print_units("Manual"), ("P",), group="Mission 5")

        # ----- Mission 6: fixed timestep + trails (no conflict with M3) -----
        if type(m) is Mission6FixedTimestep:
            self.kb.add("Trails toggle (M6)", lambda: m.toggle_trails(), ("B",), group="Mission 6")
            self.kb.add("Trails clear (M6)",  lambda: m.clear_trails(),  ("U",), group="Mission 6")
            # Trail length: J/K (+ PgDn/PgUp aliases)
            self.kb.add("Trail length shorter (M6)", lambda: m.decrease_trail_len(),
                        ("J","KEY_J"), ("PAGEDOWN","PAGE_DOWN"), group="Mission 6")
            self.kb.add("Trail length longer (M6)",  lambda: m.increase_trail_len(),
                        ("K","KEY_K"), ("PAGEUP","PAGE_UP"), group="Mission 6")
            # Trail point size on 9/0 (чтобы не пересекаться с , . из М3)
            self.kb.add("Trail point size - (M6)",   lambda: setattr(m, "trail_point_size_px",
                                                                    max(1.0, float(getattr(m,"trail_point_size_px",4.0))/1.2)),
                        ("_9","NUM_9","NUMBER_9","NINE","N9","K9"), group="Mission 6")
            self.kb.add("Trail point size + (M6)",   lambda: setattr(m, "trail_point_size_px",
                                                                    min(64.0, float(getattr(m,"trail_point_size_px",4.0))*1.2)),
                        ("_0","NUM_0","NUMBER_0","ZERO","N0","K0"), group="Mission 6")

        # ----- Mission 7: light bending (Schwarzschild geodesics) -----
        if isinstance(m, Mission7LightBending):
            
            self.kb.add("Trails toggle (M7)", lambda: m.toggle_trails(), ("B",), group="Mission 7")
            self.kb.add("Trails clear (M7)",  lambda: m.clear_trails(),  ("U",), group="Mission 7")
            self.kb.add("Trail length shorter (M7)", lambda: m.decrease_trail_len(),
                        ("J","KEY_J"), ("PAGEDOWN","PAGE_DOWN"), group="Mission 7")
            self.kb.add("Trail length longer (M7)",  lambda: m.increase_trail_len(),
                        ("K","KEY_K"), ("PAGEUP","PAGE_UP"), group="Mission 7")
            self.kb.add("Trail point size - (M7)",   lambda: setattr(m, "trail_point_size_px",
                                                                    max(1.0, float(getattr(m,"trail_point_size_px",4.0))/1.2)),
                        ("_9","NUM_9","NUMBER_9","NINE","N9","K9"), group="Mission 7")
            self.kb.add("Trail point size + (M7)",   lambda: setattr(m, "trail_point_size_px",
                                                                    min(64.0, float(getattr(m,"trail_point_size_px",4.0))*1.2)),
                        ("_0","NUM_0","NUMBER_0","ZERO","N0","K0"), group="Mission 7")

            def _m7_reset():
                print("[M7] Reseed geodesics with current φ-window (no re-init)")
                if hasattr(m, "_reseed_geodesics"):
                    m._reseed_geodesics(keep_trails=False)

            def _phi_slower():
                m.phi_rate = max(0.05, float(getattr(m, "phi_rate", 1.5)) / 1.2)
                print(f"[M7] Angular speed φ̇ slower → {m.phi_rate:.4f} rad/s")

            def _phi_faster():
                m.phi_rate = min(10.0, float(getattr(m, "phi_rate", 1.5)) * 1.2)
                print(f"[M7] Angular speed φ̇ faster → {m.phi_rate:.4f} rad/s")

            def _window_narrow():
                m.phi0    *= 0.90
                m.phi_max *= 0.90
                print(f"[M7] Angle window narrower → [{m.phi0:.3f}, {m.phi_max:.3f}] rad")
                _m7_reset()

            def _window_widen():
                m.phi0    *= 1.10
                m.phi_max *= 1.10
                print(f"[M7] Angle window wider   → [{m.phi0:.3f}, {m.phi_max:.3f}] rad")
                _m7_reset()

            def _toggle_loop():
                m.set_loop(not getattr(m, "loop_rays", True))
                print(f"[M7] Loop rays: {m.loop_rays}")

            self.kb.add("M7: Reset geodesics",        _m7_reset,     ("R",), group="Mission 7")
            self.kb.add("M7: Angular speed φ̇ slower", _phi_slower,   ("_1","NUM_1","NUMBER_1","ONE"), group="Mission 7")
            self.kb.add("M7: Angular speed φ̇ faster", _phi_faster,   ("_2","NUM_2","NUMBER_2","TWO"), group="Mission 7")
            self.kb.add("M7: Angle window narrower",  _window_narrow,
            ("COMMA","OEM_COMMA"), group="Mission 7")        # < (Shift + ,)
            self.kb.add("M7: Angle window wider",     _window_widen,
            ("PERIOD","OEM_PERIOD"), group="Mission 7")      # > (Shift + .) 
            self.kb.add("M7: Loop rays toggle",       _toggle_loop,   ("L",), group="Mission 7")
            #self.kb.add("M7: Angle window narrower",  _window_narrow, ("SEMICOLON","OEM_1"), group="Mission 7")  # ;
            #self.kb.add("M7: Angle window wider",     _window_widen,  ("APOSTROPHE", "OEM_7", "QUOTE", "APOSTROPHE_QUOTE", "SINGLE_QUOTE", "APOSTROPHE_QUOTE"), group="Mission 7") # '

    def on_render(self, time: float, frame_time: float):
        if not self.active_mission:
            return
        self.active_mission.update(frame_time)
        self.ctx.clear(0.03, 0.04, 0.07, 1.0)
        self.ctx.screen.use()
        self.active_mission.render()
        # Poll edge-triggered hotkeys (per-frame)
        self.kb.poll()
        # Poll continuous controls (smooth movement)
        self._poll_continuous_movement()

    def _poll_continuous_movement(self):
        """
        Continuous movement of the black hole center while keys are held down.
        Kept outside KeyBinder because it's not edge-triggered.
        """
        if not self.active_mission:
            return
        m = self.active_mission
        keys = self.wnd.keys

        def pressed(name):
            k = getattr(keys, name, None)
            return bool(k) and self.wnd.is_key_pressed(k)

        step = 400.0 / 60.0  # ~400 px/s assuming ~60 fps

        if pressed("UP") or pressed("W"):
            m.center[1] = min(m.height, m.center[1] + step)
        if pressed("DOWN") or pressed("S"):
            m.center[1] = max(0.0, m.center[1] - step)
        if pressed("LEFT") or pressed("A"):
            m.center[0] = max(0.0, m.center[0] - step)
        if pressed("RIGHT") or pressed("D"):
            m.center[0] = min(m.width, m.center[0] + step)


    def key_event(self, key, action, modifiers):
        # DEBUG: see if window receives events at all
        key_name = getattr(key, "name", None) if not isinstance(key, str) else key
        print(f"[MC] key={key} name={key_name} action={action} mods={modifiers}")

        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key == keys.ESCAPE:
            self.wnd.close()
            return
        if self.active_mission:
            self.active_mission.handle_key(key, action, modifiers, keys)

    def _print_help(self):
        print(self.kb.help_text())

def prompt_for_mission():
    print("\nSelect a mission to run:")
    print("  1: Grid + Black Hole")
    print("  2: Grid + Single Light Beam")
    print("  3: Grid + Multiple Light Beams")
    print("  4: Grid + Multiple Beams + Collision Detection")
    print("  5: SI Units + Schwarzschild Radius")
    print("  6: Fixed Timestep Physics Loop")
    print("  7: Light Bending Simulation")
    print("  8: Validation Suite")
    print("  9: Red Shift Simulation")
    print("  0: Exit")
    while True:
        try:
            mission_number = int(input("Enter mission number (0-9): "))
            if 0 <= mission_number <= 9:
                return mission_number
            else:
                print("Please enter a number between 0 and 9.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    while True:
        mission_number = prompt_for_mission()
        if mission_number == 0:
            print("Exiting Mission Control. Goodbye!")
            break
        MissionControl.selected_mission_number = mission_number
        mglw.run_window_config(MissionControl)
