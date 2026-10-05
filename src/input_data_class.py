"""
Data classes for user-defined input
Children data classes are wrapped in a parent data class `InputData`
"""

import multiprocessing as mp
import os
import warnings
from dataclasses import dataclass, field
from bidict import bidict
from pyomo.environ import SolverFactory


@dataclass
class MissionParameters:
    """Data class containing mission parameters

    Args:
        n_mis: Number of missions
        n_sc_design: Number of SC design
        n_sc_per_design: Number of SC per design
        t_mis_tot: Total single mission duration, days
        t_mis: Lunar surface mission duration, days
        n_crew: Number of crew needed on lunar surface
        sample_mass: Sample collected from lunar surface, kg
            can be a list of float if value is different for each mission
        habit_pl_mass: Habitat and payload mass to lunar surface, kg
            can be a list of float if value is different for each mission
        consumption_cost: Consumption cost (food+water+oxygen), kg/(day*person)
            Defaults to 8.655 (food=1.105, water=6.37, oxygen=1.14)
        water_consumption: Water share of consumption_cost, kg/(day*person)
            Defaults to 6.37 (the water term of the default consumption_cost)
        water_recovery_rate: Fraction of water consumption recovered by a
            regenerative life support loop in the lunar surface habitat
            (urine and humidity condensate processing), 0 to 1. Applied to the
            surface consumption demand only; in-transit consumption is
            unchanged. Defaults to 0 (open loop, all water shipped), which
            leaves results unchanged. For reference, the ISS water loop
            reports roughly 0.93 to 0.98 total recovery.
        maintenance_cost: Maintenance cost per flight
            Defualts to 0.01 (1% mass per flight)
        use_increased_pl (optional): True if increased demand is used.
            Defaults to False
    """

    n_mis: int
    n_sc_design: int
    n_sc_per_design: int
    t_mis_tot: float
    t_surf_mis: float
    n_crew: int
    sample_mass: float | list[float]
    habit_pl_mass: float | list[float]
    consumption_cost: float = 8.655
    water_consumption: float = 6.37
    water_recovery_rate: float = 0.0
    maintenance_cost: float = 0.01
    time_interval: int = 365
    use_increased_pl: bool = False
    increased_pl_factor: float = 1.5

    def __post_init__(self):
        """Sanity check for input values"""

        assert all(
            value > 0
            for value in [
                self.n_mis,
                self.n_sc_design,
                self.n_sc_per_design,
                self.t_mis_tot,
                self.t_surf_mis,
                self.n_crew,
                self.consumption_cost,
                self.maintenance_cost,
                self.time_interval,
            ]
        ), """
        Error:
        All of the following must be positive.
        Received values:
            Numer of mission: {}
            Number of SC design: {}
            Number of SC per design: {}
            Total single mission duration: {}
            Lunar surface mission duration: {}
            Number of crew: {}
            consumption cost: {}
            Maintenance cost: {}
            Mission time interval: {}
        """.format(
            self.n_mis,
            self.n_sc_design,
            self.n_sc_per_design,
            self.t_mis_tot,
            self.t_surf_mis,
            self.n_crew,
            self.consumption_cost,
            self.maintenance_cost,
            self.time_interval,
        )

        assert 0 <= self.water_recovery_rate <= 1, """
        Water recovery rate must be between 0 and 1. Received: {}""".format(
            self.water_recovery_rate
        )
        assert 0 <= self.water_consumption <= self.consumption_cost, """
        Water consumption must be between 0 and the consumption cost.
        Received water consumption: {}, consumption cost: {}""".format(
            self.water_consumption, self.consumption_cost
        )

        if isinstance(self.sample_mass, list):
            assert len(self.sample_mass) == self.n_mis, """
            Size of sample mass list must be the same as the number of missions."""
            assert all(value >= 0 for value in self.sample_mass)
            self.sample_mass_ls = self.sample_mass
        else:
            assert self.sample_mass >= 0
            self.sample_mass_ls = [self.sample_mass] * self.n_mis

        if isinstance(self.habit_pl_mass, list):
            assert len(self.habit_pl_mass) == self.n_mis, """
            Size of habitat+payload mass list must be the same as the number of missions."""
            assert all(value >= 0 for value in self.habit_pl_mass)
            self.habit_pl_mass_ls = self.habit_pl_mass
        else:
            assert self.habit_pl_mass >= 0
            self.habit_pl_mass_ls = [self.habit_pl_mass] * self.n_mis

    @property
    def net_consumption_cost(self) -> float:
        """Surface consumption that must be delivered, kg/(day*person), after water recovery"""
        return self.consumption_cost - self.water_consumption * self.water_recovery_rate


@dataclass
class SCParameters:
    """Data class containing SC data

    Args:
        isp: specific impulse
        oxi_fuel_ratio: SC propellant oxidizer to fuel ratio
        prop_density: SC propellant density, kg/m^3
        misc_mass_factor: SC misc. mass fraction; higher = conservative
        var_names (optional): list of spacecraft attribute/variable names
        var_ub (optional): list of spacecraft attribute/variable upper bounds
        aggressive_SC_design (optional): True if aggressive sizing model is used
        g0 (optional): standard gravitational acceleration, m/s^2
    """

    isp: float
    oxi_fuel_ratio: float
    prop_density: float
    misc_mass_fraction: float
    var_names: list[str] = field(
        default_factory=lambda: ["payload", "propellant", "dry mass"]
    )
    var_lb: list[float] = field(default_factory=lambda: [500, 1000, 4000])
    var_ub: list[float] = field(default_factory=lambda: [10000, 300000, 100000])
    aggressive_SC_design: bool = False
    g0: float = 9.8

    def __post_init__(self):
        """Sanity check for input values"""

        assert all(
            value > 0
            for value in [
                self.isp,
                self.oxi_fuel_ratio,
                self.prop_density,
                self.misc_mass_fraction,
            ]
        ), """
        Error:
        All of the following must be positive.
        Received values:
            Specific impulse: {}
            Oxidizer to fuel ratio: {}
            Propellant density: {}
            Misc. mass fraction: {}
        """.format(
            self.isp,
            self.oxi_fuel_ratio,
            self.prop_density,
            self.misc_mass_fraction,
        )
        assert all(value > 0 for value in self.var_ub), """
        All spacecraft variable upper bounds must be positive."""
        assert len(self.var_names) == len(self.var_ub), """
        Number of spacecraft variable names and their lower bounds must be the same."""
        assert 0 <= self.misc_mass_fraction < 1, """
        Misc. mass fraction must be in (0, 1]. Received value: {}
        """.format(self.misc_mass_fraction)

        self.oxi_prop_ratio: float = 1 / (1 + self.oxi_fuel_ratio)
        self.fuel_prop_ratio: float = 1 - self.oxi_prop_ratio
        self.n_sc_vars: int = len(self.var_names)


@dataclass(frozen=True)
class ISRUParameters:
    """Data class containing ISRU data
    Note that H2O produces LO2/LH2 + extra O2

    Args:
        use_isru: True if ISRU is used
        n_isru_design: Number of ISRU design
        H2_H2O_ratio: H2 production per H2O
        O2_H2O_ratio: O2 production per H2O
        production_rate: ISRU production rate, production[kg] per year and per ISRU mass[kg]
        decay_rate: ISRU productivity decay rate per year
        maintenance_cost: ISRU maintenance cost, cost[kg] per year and per ISRU mass[kg]
    """

    use_isru: bool
    n_isru_design: int
    H2_H2O_ratio: float
    O2_H2O_ratio: float
    production_rate: float
    decay_rate: float
    maintenance_cost: float
    n_isru_vars: int = 1

    def __post_init__(self):
        """Sanity check for input values"""

        if self.use_isru:
            assert all(
                value > 0 for value in [self.n_isru_design, self.n_isru_vars]
            ), """
            Number of ISRU design and variable per design must be positive.
            Received value:
                Number of ISRU design: {}
                Number of varaibles per design: {}
            """.format(self.n_isru_design, self.n_isru_vars)

        assert self.production_rate > 0, """
        ISRU production rate must be positive. Received value: {}
        """.format(self.production_rate)

        assert abs(self.H2_H2O_ratio + self.O2_H2O_ratio) <= 1 + 1e-6, """
        ISRU H2 and O2 production must sum up to 1."""

        assert all(
            0 < value < 1
            for value in [
                self.H2_H2O_ratio,
                self.O2_H2O_ratio,
                self.decay_rate,
                self.maintenance_cost,
            ]
        ), """
        Error:
        All of the following must be in (0,1).
        Received values:
            H2 to H2O ratio: {}
            O2 to H2O ratio: {}
            ISRU decay rate: {}
            ISRU maintenance cost: {}
        """.format(
            self.H2_H2O_ratio,
            self.O2_H2O_ratio,
            self.decay_rate,
            self.maintenance_cost,
        )


@dataclass(frozen=True)
class ALCParameters:
    """Data class containing ALC parameters

    Args:
        initial_weight_coefficient: ALC parameter
        weight_update_coefficient: ALC parameter
        weight_update_fraction: ALC parameter
        tol_outer: ALC outer loop tolerance
        tol_inner: ALC inner loop tolerance
        parallel_mode (optional): True if subproblems solved in parallel. Defaults to False.
    """

    initial_weight: float
    initial_weight_coefficient: float
    weight_update_coefficient: float
    weight_update_fraction: float
    tol_outer: float
    tol_inner: float
    prioritized_var_name: str | None = None
    use_admm: bool = False
    parallel_mode: bool = False
    update_initial_weight: bool = False

    def __post_init__(self):
        """Sanity check for input values"""

        assert self.initial_weight > 0, """
        Initial weight must be positive. Received value: {}
        """.format(self.initial_weight)

        assert 1e-3 < self.initial_weight_coefficient < 1, """
        Initial weight coefficient must be in (1e-3, 1). Received value: {}
        """.format(self.initial_weight_coefficient)

        assert self.weight_update_coefficient > 1, """
        Weight update coefficient must be greater than 1.
        Recommended range is (2,3). Received value: {}
        """.format(self.weight_update_coefficient)

        assert 0 < self.weight_update_fraction < 1, """
        Weight update fraction must be in (0,1). Recommended value is 0.25.
        Received value: {}""".format(self.weight_update_fraction)


@dataclass
class CommodityDetails:
    """Data class containing commodity details
    Args:
        int_com_names: List of integer commodity names
        int_com_costs: List of integer commodity costs per unit
        cnt_com_names: List of continuous commodity names
        prop_com_names (optional): List of propellant commodity names
        com_names_w_unlim_earth_supply (optional): List of commodity names
            with unlimited supply from Earth node.
            Do NOT include any commodity that needs to return to Earth
            at the end of each mission (e.g., sample, crew).
    """

    int_com_names: list[str]
    int_com_costs: list[float]
    cnt_com_names: list[str]
    prop_com_names: list[str] = field(default_factory=lambda: ["oxygen", "hydrogen"])
    com_names_w_unlim_earth_supply: list[str] = field(
        default_factory=lambda: [
            "plant",
            "maintenance",
            "consumption",
            "habitat",
            "oxygen",
            "hydrogen",
        ]
    )

    def __post_init__(self):
        """Sanity check for input values and define derived variables"""

        assert len(self.int_com_names) == len(self.int_com_costs), """
        Number of integer commodity names and costs must be the same."""

        assert all(cost > 0 for cost in self.int_com_costs), """
        All commodity costs must be positive."""

        assert all(prop in self.cnt_com_names for prop in self.prop_com_names), """
        All propellant commodity names must be
        in the continuous commodity names list."""

        assert all(
            com in self.int_com_names + self.cnt_com_names
            for com in self.com_names_w_unlim_earth_supply
        ), """
        All commodity names with unlimited supply from Earth
        must be in the commodity names list."""

        self.n_int_com: int = len(self.int_com_names)
        self.n_cnt_com: int = len(self.cnt_com_names)
        self.n_com: int = self.n_int_com + self.n_cnt_com
        self.cnt_com_costs: list[float] = [1.0] * self.n_cnt_com
        self.com_names: list[str] = self.int_com_names + self.cnt_com_names
        self.com_costs: list[float] = self.int_com_costs + self.cnt_com_costs
        self.nonprop_com_names: list[str] = [
            com for com in self.com_names if com not in self.prop_com_names
        ]


@dataclass
class NodeDetails:
    """Data class containing node details

    Args:
        node_names: List of node names
        is_path_garph (optional): True if the defined graph is a path graph
            (a graph with only one path like o-o-o-o). Defaults to True.
        outbound_path (optional): Sequence of nodes from source node
            to desitnation, in terms of node names.
            Only needed if the graph is a path graph.
        holdover_nodes (optional): Set of nodes where holdover arcs are allowed
        inbound_path (optional): Sequence of nodes from destination to source,
            in terms of node names. If not specified, reverse of outboud is assumed.
    """

    node_names: list[str]
    is_path_graph: bool = True
    outbound_path: list[str] = field(
        default_factory=lambda: ["Earth", "LEO", "LLO", "LS"]
    )
    inbound_path: list[str] | None = None
    holdover_nodes: list[str] = field(default_factory=lambda: ["LLO", "LS"])
    source_node: str | None = None
    destination_node: str | None = None

    def __post_init__(self):
        self.n_nodes = len(self.node_names)

        if self.is_path_graph:
            assert all(node in self.node_names for node in self.outbound_path), """
            One or more nodes in the specified outbound path
            (sequense of nodes from source to destination) cannot be found in
            the node name list. If the graph is not a path graph,
            set is_path_graph to False."""
            assert all(node in self.outbound_path for node in self.node_names), """
            Not all nodes appear in the specified outbound path
            (sequense of nodes from source to destination).
            If the graph is not a path graph, set is_path_graph to False."""
            for node in self.holdover_nodes:
                assert node in self.node_names, """
                Node {} in holdover nodes is not in the defined set of nodes.""".format(
                    node
                )
            if self.inbound_path:
                assert self.inbound_path == self.outbound_path[::-1]
            else:
                self.inbound_path = self.outbound_path[::-1]
            if not self.source_node:
                self.source_node = self.outbound_path[0]
            else:
                assert self.source_node in self.node_names, """
                The specified source node is not in the defined set of nodes."""
            if not self.destination_node:
                self.destination_node = self.outbound_path[-1]
            else:
                assert self.destination_node in self.node_names, """
                The specified destination node is not in the defined set of nodes."""
        else:
            warnings.warn(
                """The specified graph is not a path graph.
                Some features may be limited, especially in the output file"""
            )


@dataclass
class ScenarioDistribution:
    """Data class containing scnenario info and distribution for 2SSP

    Args:
        sample_mass_2nd: Sample mass collected from LS for each scenario of 2nd mission
        habit_pl_mass_2nd: Habitat and payload mass for each scenario of 2nd mission
        scenario_prob(optional): List of scenario probabilities.
            Defaults to equal probability.
    """

    sample_mass_2nd: list[float]
    habit_pl_mass_2nd: list[float]
    scenario_prob: list[float] = field(default_factory=lambda: [])

    def __post_init__(self):
        assert len(self.sample_mass_2nd) == len(self.habit_pl_mass_2nd), """
        Size of sample mass distribution and
        habitat/payload mass distribution must be the same."""
        assert all(
            value >= 0 for value in self.sample_mass_2nd + self.habit_pl_mass_2nd
        ), """
        All values in sample and habitat mass must be nonnegative.
        """
        self.n_scenarios: int = len(self.sample_mass_2nd)
        if self.scenario_prob:
            assert len(self.scenario_prob) == len(self.sample_mass_2nd), """
            Size of scenarios probability must be the same as the other distributions."""
            assert sum(self.scenario_prob) == 1, """
            Sum of scenario probabilities must be 1."""
            assert all(value > 0 and value < 1 for value in self.scenario_prob), """
                All scenario probabilities must be in (0, 1).
                """
        else:
            self.scenario_prob = [1 / self.n_scenarios] * self.n_scenarios


@dataclass
class RuntimeSettings:
    """Data class containing code settings

    Args:
        pwl_increment_list: List of PWL increments to try
        store_results_to_csv(optional): True if results stored to a .csv file. Defaults to False.
        mip_solver(optional): MIP solver name. Defaults to "gurobi"
        mip_subsolver(optional): MIP subsolver name for spatial branch & bounds. Defaults to "cplex"
        max_time(optional): maximum computation time in seconds. Defaults to 1000
        max_time_wo_imprv(optional): maximum computation time without improvement in seconds.
            Used for Baron and defaults to 1000.
        max_threads(optional): maximum number of CPU threads allowed to use for computation
        solver_verbose(optional): True if solver output is needed on terminal. Defaults to False.
        keep_files: True if misc. solver output files (log file,
            solution file, and model file) are kept. Defaults to False.
        cplex_path(optional): Abosolute path to CPLEX executable (libcplex****.so). Relative path is not supported.
    """

    pwl_increment_list: list[float]
    store_results_to_csv: bool = False
    mip_solver: str = field(default_factory=lambda: "gurobi")
    mip_subsolver: str = field(default_factory=lambda: "cplex")
    max_time: float = 1000
    max_time_wo_imprv: float = 1000
    max_threads: int = mp.cpu_count()
    solver_verbose: bool = False
    keep_files: bool = False
    cplex_path: str = field(
        default_factory=lambda: (
            "/home/masafumi/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/libcplex2211.so"
        )
    )

    def __post_init__(self):
        if self.mip_solver in ["gurobi", "Gurobi", "GUROBI"]:
            self.mip_solver = "gurobi"
        elif self.mip_solver in ["cplex", "CPLEX", "Cplex"]:
            self.mip_solver = "cplex"
        elif self.mip_solver in ["Baron", "baron"]:
            self.mip_solver = "baron"
            if self.mip_subsolver in ["cplex", "CPLEX", "Cplex"]:
                self.mip_subsolver = "cplex"
            else:
                warnings.warn("""
                Specified MIP subsolver is not compatible with Baron or already the default subsolver of Baron.
                Proceeding with the default MIP subsolver.""")
        elif not SolverFactory(self.mip_solver).available():
            raise ValueError(
                """Invalid MIP solver.
                Please make sure the specified solver is supported by pyomo,
                and the solver is available in your environment."""
            )

        if self.mip_solver == "baron" and not os.path.exists(self.cplex_path):
            print(
                """CPLEX path is not found - unless the environmental variables are configured for CPLEX,
                another subsolver will be used for Baron """
            )

        assert self.max_threads <= mp.cpu_count(), """
        Maximum number of threads must be less than
        or equal to the number of CPU threads."""
        assert all(value > 0 for value in [self.max_time, self.max_threads]), """
        Maximum time and thread count must be positive."""


@dataclass
class InputData:
    """Parent data class containing all input parameters"""

    mission: MissionParameters
    alc: ALCParameters
    sc: SCParameters
    isru: ISRUParameters
    comdty: CommodityDetails
    node: NodeDetails
    runtime: RuntimeSettings
    scenario: ScenarioDistribution | None = None

    def __post_init__(self):
        self._create_bidicts()
        if self.alc.prioritized_var_name:
            assert self.alc.prioritized_var_name in self.sc.var_names, """
            prioritized variable name is not valid"""

        self.n_scenarios: int = 1
        self.is_stochastic: bool = False
        self.scenario_prob: list[float] = [1]
        if self.scenario:
            self.has_scenario_info: bool = True
        else:
            self.has_scenario_info: bool = False
        if self.scenario and self.mission.n_mis > 2:
            raise ValueError(
                "Stochastic optimization is only supported up to two stages."
            )

    def activate_stochasticity(self):
        if not self.scenario:
            raise ValueError("Scenario information is missing.")
        self.n_scenarios: int = self.scenario.n_scenarios
        self.is_stochastic: bool = True
        self.scenario_prob: list[float] = self.scenario.scenario_prob

    def _create_bidicts(self):
        """Create bidirectional dictionaries (key <-> attribute)

        This is for avoiding hardcoding of indices.
        The indicies can be extracted by names, and vice veersa.
        E.g., dict['out'] -> 0, dict[0] -> 'out'
        Bidicts created for integer and continusous commodity,
        total commodity (integer and continuous commodity with shared indices),
        node, flow in/out, SC variable/attribute
        """
        com_dict: dict[str, int] = {}
        int_com_dict: dict[str, int] = {}
        cnt_com_dict: dict[str, int] = {}

        for com_id in range(self.comdty.n_int_com):
            int_com_name = self.comdty.int_com_names[com_id]
            com_dict[int_com_name] = com_id
            int_com_dict[int_com_name] = com_id

        for com_id in range(self.comdty.n_cnt_com):
            cnt_com_name = self.comdty.cnt_com_names[com_id]
            com_dict[cnt_com_name] = com_id + self.comdty.n_int_com
            cnt_com_dict[cnt_com_name] = com_id

        node_dict: dict[str, int] = {}
        for node_id in range(self.node.n_nodes):
            node_name = self.node.node_names[node_id]
            node_dict[node_name] = node_id

        flow_dict: dict[str, int] = {"out": 0, "in": 1}

        sc_var_dict: dict[str, int] = {}
        for sc_var_id in range(self.sc.n_sc_vars):
            sc_var_name = self.sc.var_names[sc_var_id]
            sc_var_dict[sc_var_name] = sc_var_id

        # make dictionaries bidriectional and store them as attributes
        self.com_dict = bidict(com_dict)
        self.int_com_dict = bidict(int_com_dict)
        self.cnt_com_dict = bidict(cnt_com_dict)
        self.node_dict = bidict(node_dict)
        self.flow_dict = bidict(flow_dict)
        self.sc_var_dict = bidict(sc_var_dict)
