"""Build a large physics hypergraph spanning many chapters.

Each chapter is a hypernym that groups physics variables (its hyponyms).
A hyperedge is a formula: it connects a set of input variables to one output
variable, and is tagged with its chapter (its hypernym).
"""
import json
from pathlib import Path

# (chapter, output, [inputs], "human-readable formula", output_unit)
FORMULAS = [
    # ---------------- Kinematics ----------------
    ("kinematics", "velocity", ["displacement", "time"], "displacement / time", "m/s"),
    ("kinematics", "displacement", ["velocity", "time"], "velocity * time", "m"),
    ("kinematics", "time", ["displacement", "velocity"], "displacement / velocity", "s"),
    ("kinematics", "speed", ["distance", "time"], "distance / time", "m/s"),
    ("kinematics", "distance", ["speed", "time"], "speed * time", "m"),
    ("kinematics", "acceleration", ["final_velocity", "initial_velocity", "time"],
     "(final_velocity - initial_velocity) / time", "m/s^2"),
    ("kinematics", "final_velocity", ["initial_velocity", "acceleration", "time"],
     "initial_velocity + acceleration * time", "m/s"),
    ("kinematics", "displacement", ["initial_velocity", "acceleration", "time"],
     "initial_velocity * time + 0.5 * acceleration * time^2", "m"),
    ("kinematics", "final_velocity", ["initial_velocity", "acceleration", "displacement"],
     "sqrt(initial_velocity^2 + 2 * acceleration * displacement)", "m/s"),
    ("kinematics", "average_velocity", ["initial_velocity", "final_velocity"],
     "(initial_velocity + final_velocity) / 2", "m/s"),
    ("kinematics", "displacement", ["average_velocity", "time"], "average_velocity * time", "m"),
    ("kinematics", "horizontal_velocity", ["initial_velocity", "angle"],
     "initial_velocity * cos(angle)", "m/s"),
    ("kinematics", "vertical_velocity", ["initial_velocity", "angle"],
     "initial_velocity * sin(angle)", "m/s"),
    ("kinematics", "range", ["initial_velocity", "angle", "gravity"],
     "(initial_velocity^2 * sin(2*angle)) / gravity", "m"),
    ("kinematics", "max_height", ["initial_velocity", "angle", "gravity"],
     "(initial_velocity * sin(angle))^2 / (2 * gravity)", "m"),
    ("kinematics", "time_of_flight", ["initial_velocity", "angle", "gravity"],
     "2 * initial_velocity * sin(angle) / gravity", "s"),
    ("kinematics", "horizontal_displacement", ["horizontal_velocity", "time"],
     "horizontal_velocity * time", "m"),
    ("kinematics", "vertical_displacement", ["vertical_velocity", "gravity", "time"],
     "vertical_velocity * time - 0.5 * gravity * time^2", "m"),

    # ---------------- Dynamics ----------------
    ("dynamics", "force", ["mass", "acceleration"], "mass * acceleration", "N"),
    ("dynamics", "mass", ["force", "acceleration"], "force / acceleration", "kg"),
    ("dynamics", "acceleration", ["force", "mass"], "force / mass", "m/s^2"),
    ("dynamics", "weight", ["mass", "gravity"], "mass * gravity", "N"),
    ("dynamics", "normal_force", ["mass", "gravity"], "mass * gravity", "N"),
    ("dynamics", "friction_force", ["coefficient_of_friction", "normal_force"],
     "mu * normal_force", "N"),
    ("dynamics", "net_force", ["applied_force", "friction_force"],
     "applied_force - friction_force", "N"),
    ("dynamics", "momentum", ["mass", "velocity"], "mass * velocity", "kg*m/s"),
    ("dynamics", "impulse", ["force", "time"], "force * time", "kg*m/s"),
    ("dynamics", "impulse", ["final_momentum", "initial_momentum"],
     "final_momentum - initial_momentum", "kg*m/s"),
    ("dynamics", "change_in_momentum", ["mass", "final_velocity", "initial_velocity"],
     "mass * (final_velocity - initial_velocity)", "kg*m/s"),
    ("dynamics", "centripetal_force", ["mass", "velocity", "radius"],
     "mass * velocity^2 / radius", "N"),
    ("dynamics", "centripetal_acceleration", ["velocity", "radius"],
     "velocity^2 / radius", "m/s^2"),
    ("dynamics", "centripetal_acceleration", ["angular_velocity", "radius"],
     "angular_velocity^2 * radius", "m/s^2"),
    ("dynamics", "tension", ["mass", "gravity", "acceleration"],
     "mass * (gravity + acceleration)", "N"),
    ("dynamics", "spring_force", ["spring_constant", "displacement"],
     "spring_constant * displacement", "N"),
    ("dynamics", "pressure", ["force", "area"], "force / area", "Pa"),

    # ---------------- Energy / Work / Power ----------------
    ("energy", "kinetic_energy", ["mass", "velocity"], "0.5 * mass * velocity^2", "J"),
    ("energy", "potential_energy", ["mass", "gravity", "height"],
     "mass * gravity * height", "J"),
    ("energy", "elastic_pe", ["spring_constant", "displacement"],
     "0.5 * spring_constant * displacement^2", "J"),
    ("energy", "work", ["force", "displacement"], "force * displacement", "J"),
    ("energy", "work", ["force", "displacement", "angle"],
     "force * displacement * cos(angle)", "J"),
    ("energy", "power", ["work", "time"], "work / time", "W"),
    ("energy", "power", ["force", "velocity"], "force * velocity", "W"),
    ("energy", "mechanical_energy", ["kinetic_energy", "potential_energy"],
     "kinetic_energy + potential_energy", "J"),
    ("energy", "efficiency", ["useful_output", "total_input"],
     "useful_output / total_input", ""),
    ("energy", "change_in_kinetic_energy", ["work"], "work", "J"),
    ("energy", "height", ["potential_energy", "mass", "gravity"],
     "potential_energy / (mass * gravity)", "m"),
    ("energy", "velocity", ["kinetic_energy", "mass"], "sqrt(2 * kinetic_energy / mass)", "m/s"),

    # ---------------- Rotational ----------------
    ("rotational", "angular_velocity", ["angular_displacement", "time"],
     "angular_displacement / time", "rad/s"),
    ("rotational", "angular_acceleration",
     ["final_angular_velocity", "initial_angular_velocity", "time"],
     "(final_angular_velocity - initial_angular_velocity) / time", "rad/s^2"),
    ("rotational", "angular_displacement",
     ["initial_angular_velocity", "angular_acceleration", "time"],
     "w0*t + 0.5*alpha*t^2", "rad"),
    ("rotational", "velocity", ["angular_velocity", "radius"],
     "angular_velocity * radius", "m/s"),
    ("rotational", "tangential_acceleration", ["angular_acceleration", "radius"],
     "angular_acceleration * radius", "m/s^2"),
    ("rotational", "torque", ["force", "radius"], "force * radius", "N*m"),
    ("rotational", "torque", ["force", "radius", "angle"],
     "force * radius * sin(angle)", "N*m"),
    ("rotational", "torque", ["moment_of_inertia", "angular_acceleration"],
     "moment_of_inertia * angular_acceleration", "N*m"),
    ("rotational", "angular_momentum", ["moment_of_inertia", "angular_velocity"],
     "moment_of_inertia * angular_velocity", "kg*m^2/s"),
    ("rotational", "moment_of_inertia", ["mass", "radius"], "mass * radius^2", "kg*m^2"),
    ("rotational", "rotational_kinetic_energy", ["moment_of_inertia", "angular_velocity"],
     "0.5 * moment_of_inertia * angular_velocity^2", "J"),
    ("rotational", "frequency", ["period"], "1 / period", "Hz"),
    ("rotational", "angular_velocity", ["frequency"], "2*pi*frequency", "rad/s"),
    ("rotational", "period", ["angular_velocity"], "2*pi / angular_velocity", "s"),

    # ---------------- SHM ----------------
    ("shm", "period", ["mass", "spring_constant"],
     "2*pi*sqrt(mass / spring_constant)", "s"),
    ("shm", "period_pendulum", ["length", "gravity"],
     "2*pi*sqrt(length / gravity)", "s"),
    ("shm", "angular_frequency", ["spring_constant", "mass"],
     "sqrt(spring_constant / mass)", "rad/s"),
    ("shm", "displacement", ["amplitude", "angular_frequency", "time"],
     "amplitude * cos(angular_frequency * time)", "m"),
    ("shm", "max_velocity", ["amplitude", "angular_frequency"],
     "amplitude * angular_frequency", "m/s"),
    ("shm", "max_acceleration", ["amplitude", "angular_frequency"],
     "amplitude * angular_frequency^2", "m/s^2"),
    ("shm", "total_energy_shm", ["spring_constant", "amplitude"],
     "0.5 * spring_constant * amplitude^2", "J"),

    # ---------------- Gravitation ----------------
    ("gravitation", "gravitational_force", ["mass1", "mass2", "distance"],
     "G * m1 * m2 / r^2", "N"),
    ("gravitation", "gravitational_pe", ["mass1", "mass2", "distance"],
     "-G * m1 * m2 / r", "J"),
    ("gravitation", "gravity", ["earth_mass", "earth_radius"],
     "G * M / R^2", "m/s^2"),
    ("gravitation", "escape_velocity", ["earth_mass", "earth_radius"],
     "sqrt(2*G*M / R)", "m/s"),
    ("gravitation", "orbital_velocity", ["earth_mass", "orbital_radius"],
     "sqrt(G*M / r)", "m/s"),
    ("gravitation", "orbital_period", ["earth_mass", "orbital_radius"],
     "2*pi*sqrt(r^3 / (G*M))", "s"),

    # ---------------- Waves & Sound ----------------
    ("waves", "wave_speed", ["frequency", "wavelength"], "frequency * wavelength", "m/s"),
    ("waves", "wavelength", ["wave_speed", "frequency"], "wave_speed / frequency", "m"),
    ("waves", "frequency", ["wave_speed", "wavelength"], "wave_speed / wavelength", "Hz"),
    ("waves", "period", ["frequency"], "1 / frequency", "s"),
    ("waves", "wave_number", ["wavelength"], "2*pi / wavelength", "1/m"),
    ("waves", "angular_frequency", ["frequency"], "2*pi * frequency", "rad/s"),
    ("waves", "speed_string", ["tension", "linear_density"],
     "sqrt(tension / linear_density)", "m/s"),
    ("waves", "intensity", ["power", "area"], "power / area", "W/m^2"),
    ("waves", "doppler_frequency",
     ["frequency", "wave_speed", "observer_velocity", "source_velocity"],
     "f*(v+vo)/(v-vs)", "Hz"),

    # ---------------- Thermodynamics ----------------
    ("thermodynamics", "pressure", ["moles", "gas_constant", "temperature", "volume"],
     "n*R*T / V", "Pa"),
    ("thermodynamics", "volume", ["moles", "gas_constant", "temperature", "pressure"],
     "n*R*T / P", "m^3"),
    ("thermodynamics", "temperature", ["pressure", "volume", "moles", "gas_constant"],
     "P*V / (n*R)", "K"),
    ("thermodynamics", "average_kinetic_energy", ["boltzmann_constant", "temperature"],
     "1.5 * k_B * temperature", "J"),
    ("thermodynamics", "heat", ["mass", "specific_heat", "change_in_temperature"],
     "mass * c * dT", "J"),
    ("thermodynamics", "work_thermo", ["pressure", "change_in_volume"],
     "pressure * dV", "J"),
    ("thermodynamics", "internal_energy_change", ["heat", "work_thermo"],
     "heat - work_thermo", "J"),
    ("thermodynamics", "efficiency_carnot", ["cold_temperature", "hot_temperature"],
     "1 - T_c / T_h", ""),

    # ---------------- Fluids ----------------
    ("fluids", "pressure", ["density", "gravity", "height"],
     "density * gravity * height", "Pa"),
    ("fluids", "buoyant_force", ["density_fluid", "gravity", "volume_displaced"],
     "rho * g * V", "N"),
    ("fluids", "density", ["mass", "volume"], "mass / volume", "kg/m^3"),
    ("fluids", "mass", ["density", "volume"], "density * volume", "kg"),
    ("fluids", "volume", ["mass", "density"], "mass / density", "m^3"),
    ("fluids", "flow_velocity", ["area1", "velocity1", "area2"],
     "A1*v1/A2", "m/s"),

    # ---------------- Electrostatics ----------------
    ("electrostatics", "coulomb_force", ["charge1", "charge2", "distance"],
     "k * q1 * q2 / r^2", "N"),
    ("electrostatics", "electric_field", ["force", "charge"], "force / charge", "V/m"),
    ("electrostatics", "electric_field", ["charge", "distance"], "k * q / r^2", "V/m"),
    ("electrostatics", "electric_potential", ["charge", "distance"], "k * q / r", "V"),
    ("electrostatics", "electric_potential_energy", ["charge1", "charge2", "distance"],
     "k * q1 * q2 / r", "J"),
    ("electrostatics", "voltage", ["work", "charge"], "work / charge", "V"),
    ("electrostatics", "capacitance", ["charge", "voltage"], "charge / voltage", "F"),
    ("electrostatics", "energy_capacitor", ["capacitance", "voltage"],
     "0.5 * capacitance * voltage^2", "J"),
    ("electrostatics", "energy_capacitor", ["charge", "voltage"],
     "0.5 * charge * voltage", "J"),

    # ---------------- Current Electricity ----------------
    ("current_electricity", "current", ["charge", "time"], "charge / time", "A"),
    ("current_electricity", "voltage", ["current", "resistance"],
     "current * resistance", "V"),
    ("current_electricity", "resistance", ["voltage", "current"],
     "voltage / current", "ohm"),
    ("current_electricity", "resistance", ["resistivity", "length", "area"],
     "rho*L / A", "ohm"),
    ("current_electricity", "power", ["voltage", "current"], "voltage * current", "W"),
    ("current_electricity", "power", ["current", "resistance"],
     "current^2 * resistance", "W"),
    ("current_electricity", "power", ["voltage", "resistance"],
     "voltage^2 / resistance", "W"),
    ("current_electricity", "energy_electric", ["power", "time"], "power * time", "J"),
    ("current_electricity", "resistance_series", ["r1", "r2"], "r1 + r2", "ohm"),
    ("current_electricity", "resistance_parallel", ["r1", "r2"],
     "r1*r2 / (r1+r2)", "ohm"),

    # ---------------- Magnetism / Induction ----------------
    ("magnetism", "magnetic_force", ["charge", "velocity", "magnetic_field"],
     "q * v * B", "N"),
    ("magnetism", "magnetic_force_wire", ["current", "length", "magnetic_field"],
     "I * L * B", "N"),
    ("magnetism", "magnetic_field_solenoid",
     ["permeability", "turns_per_length", "current"], "mu0 * n * I", "T"),
    ("magnetism", "magnetic_flux", ["magnetic_field", "area"],
     "magnetic_field * area", "Wb"),
    ("magnetism", "voltage_induced", ["change_in_flux", "time"],
     "-dPhi / dt", "V"),
    ("magnetism", "voltage_induced_motional", ["magnetic_field", "length", "velocity"],
     "B * L * v", "V"),
    ("magnetism", "emf_self", ["inductance", "change_in_current", "time"],
     "-L * dI/dt", "V"),
    ("magnetism", "energy_inductor", ["inductance", "current"],
     "0.5 * L * I^2", "J"),

    # ---------------- AC Circuits ----------------
    ("ac_circuits", "inductive_reactance", ["angular_frequency", "inductance"],
     "omega * L", "ohm"),
    ("ac_circuits", "capacitive_reactance", ["angular_frequency", "capacitance"],
     "1 / (omega * C)", "ohm"),
    ("ac_circuits", "impedance",
     ["resistance", "inductive_reactance", "capacitive_reactance"],
     "sqrt(R^2 + (XL-XC)^2)", "ohm"),
    ("ac_circuits", "voltage_rms", ["peak_voltage"], "Vp / sqrt(2)", "V"),
    ("ac_circuits", "current_rms", ["peak_current"], "Ip / sqrt(2)", "A"),
    ("ac_circuits", "power_ac", ["voltage_rms", "current_rms", "phase"],
     "Vrms*Irms*cos(phi)", "W"),

    # ---------------- Optics ----------------
    ("optics", "refractive_index", ["speed_of_light", "wave_speed"],
     "c / v", ""),
    ("optics", "wave_speed", ["speed_of_light", "refractive_index"],
     "c / n", "m/s"),
    ("optics", "image_distance", ["focal_length", "object_distance"],
     "1/(1/f - 1/do)", "m"),
    ("optics", "magnification", ["image_height", "object_height"],
     "image_height / object_height", ""),
    ("optics", "magnification", ["image_distance", "object_distance"],
     "-image_distance / object_distance", ""),
    ("optics", "fringe_width", ["wavelength", "screen_distance", "slit_separation"],
     "lambda * D / d", "m"),

    # ---------------- Modern Physics ----------------
    ("modern_physics", "photon_energy", ["planck_constant", "frequency"],
     "h * frequency", "J"),
    ("modern_physics", "photon_energy", ["planck_constant", "speed_of_light", "wavelength"],
     "h*c / wavelength", "J"),
    ("modern_physics", "de_broglie_wavelength", ["planck_constant", "momentum"],
     "h / momentum", "m"),
    ("modern_physics", "max_kinetic", ["photon_energy", "work_function"],
     "photon_energy - work_function", "J"),
    ("modern_physics", "mass_energy", ["mass", "speed_of_light"],
     "m * c^2", "J"),
    ("modern_physics", "lorentz_factor", ["velocity", "speed_of_light"],
     "1 / sqrt(1 - v^2/c^2)", ""),
]


def build():
    nodes, hypernyms = set(), {}
    for chap, out, ins, _, _ in FORMULAS:
        nodes.add(out); nodes.update(ins)
        hypernyms.setdefault(out, chap)
        for v in ins:
            hypernyms.setdefault(v, chap)
    edges = [
        {"id": f"f{i}", "domain": chap, "output": out, "inputs": ins,
         "label": label, "output_unit": unit}
        for i, (chap, out, ins, label, unit) in enumerate(FORMULAS)
    ]
    return {
        "nodes": sorted(nodes),
        "hyperedges": edges,
        "hypernyms": {v: hypernyms[v] for v in sorted(nodes)},
        "chapters": sorted({h for h in hypernyms.values()}),
    }


if __name__ == "__main__":
    g = build()
    out = Path(__file__).parent / "data" / "physics_hypergraph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(g, f, indent=2)
    print(f"Saved {out}")
    print(f"Nodes: {len(g['nodes'])}  Edges: {len(g['hyperedges'])}  "
          f"Chapters: {len(g['chapters'])}")
