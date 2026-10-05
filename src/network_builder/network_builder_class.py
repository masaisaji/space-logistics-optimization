import sys
import numpy as np
from itertools import product
from bidict import bidict

try:
    from initializer import InitMixin
    from input_data_class import InputData
except (ModuleNotFoundError, ImportError):
    sys.path.append("..")
    from initializer import InitMixin
    from input_data_class import InputData


class NetworkBuilder(InitMixin):
    """Class to build space logistic network and its properties."""

    def __init__(self, input_data: InputData) -> None:
        """
        Args:
            input_data: InputData dataclass containing data input from user
        """
        self.initialize_attributes(input_data)

        self.time_steps: list = []
        for mis in range(self.n_mis):
            self.time_steps.extend(
                [mis * self.mis.time_interval, mis * self.mis.time_interval + 1]
            )
        self._dim_time_steps = len(self.time_steps)

        # place holders
        self.int_com_demand: np.ndarray = np.zeros(
            [self.n_nodes, self.n_int_com, self._dim_time_steps, self.n_scenarios]
        )
        self.cnt_com_demand: np.ndarray = np.zeros(
            [self.n_nodes, self.n_cnt_com, self._dim_time_steps, self.n_scenarios]
        )
        self.fin_ini_mass_frac: np.ndarray = np.zeros(
            [self.n_nodes, self.n_nodes, self._dim_time_steps]
        )
        self.real_arc_time: np.ndarray = np.zeros([self.n_nodes, self.n_nodes])
        self.delta_t: np.ndarray = np.zeros(
            [self.n_nodes, self.n_nodes, self._dim_time_steps]
        )
        self.allowed_time_window: list[list] = [
            [[] for _ in range(self.n_nodes)] for _ in range(self.n_nodes)
        ]
        self.isru_work_time: np.ndarray = np.zeros([self.n_nodes, self._dim_time_steps])
        self.date_to_time_idx_dict = {}

        self._post_init()

    def _post_init(self) -> None:
        self.mis_start_dates = [n * self.mis.time_interval for n in range(self.n_mis)]
        self.mis_end_dates = [date + 1 for date in self.mis_start_dates]
        self.first_mis_time_steps: list[int] = [0, 1]
        self.second_mis_time_steps: list[int] = [
            self.mis.time_interval,
            self.mis.time_interval + 1,
        ]
        self.second_mis_start_dates: list[int] = [
            date for date in self.mis_start_dates if date in self.second_mis_time_steps
        ]
        self.second_mis_end_dates: list[int] = [
            date for date in self.mis_end_dates if date in self.second_mis_time_steps
        ]
        self.mis_time_steps: list[list[int]] = [
            self.time_steps[mis * 2 : (mis + 1) * 2] for mis in range(self.n_mis)
        ]
        if self.use_increased_pl:
            self.sample_mass_ls = [
                mass * self.mis.increased_pl_factor for mass in self.sample_mass_ls
            ]
            self.habit_pl_mass_ls = [
                mass * self.mis.increased_pl_factor for mass in self.habit_pl_mass_ls
            ]
        if self.node.is_path_graph:
            self.n_dates_until_return_mis: float = (
                sum(
                    self._get_time_of_flight(
                        self.node_dict.inverse[dep_node_id],
                        self.node_dict.inverse[dep_node_id + 1],
                    )
                    for dep_node_id in range(self.n_nodes - 1)
                )
                + self.t_surf_mis
            )
        else:
            self.n_dates_until_return_mis: float = (
                self.t_mis_tot - self.t_surf_mis
            ) / 2 + self.t_surf_mis
        self._define_date_to_time_idx_dict()
        self._set_demand()
        self._set_final_to_initial_mass_frac_for_arcs()
        self._set_actual_arc_time()
        self._set_delta_t()
        self._set_allowed_time_window()
        self._set_ISRU_work_time()

    def _get_time_of_flight(self, dep_node_name: str, arr_node_name: str) -> float:
        """get time of flight between two nodes

        Args:
            dep_node_name: name of departure node
            arr_node_name: name of arrival node
        Returns:
            float: time of flight between arrival and departure nodes in days
        """
        if set([dep_node_name, arr_node_name]) == set(["Earth", "LEO"]):
            return 1  # days
        if set([dep_node_name, arr_node_name]) == set(["LEO", "LLO"]):
            return 3  # days
        if set([dep_node_name, arr_node_name]) == set(["LLO", "LS"]):
            return 1  # days
        # TODO: implement more nodes
        else:
            NotImplementedError(
                f"Time of flight between {dep_node_name} and {arr_node_name} not implemented"
            )
            return 0

    def _get_holdover_time(self, node_name: str) -> float:
        """get holdover time at a node

        Args:
            node_name: name of node
        Returns:
            float: holdover time at the given node in days
        """
        if node_name == "LS":
            return self.t_surf_mis  # days
        if node_name == "LLO":
            # go to LS, complete mission, then come back to LLO
            return self.t_surf_mis + 2 * self._get_time_of_flight("LLO", "LS")
        if node_name == "LEO":
            return (
                self.t_surf_mis
                + 2 * self._get_time_of_flight("LLO", "LS")
                + 2 * self._get_time_of_flight("LEO", "LLO")
            )
        else:
            NotImplementedError(f"Holdover time at {node_name} is not implemented")
            return 0

    def _get_delta_v_km_s(self, dep_node_name: str, arr_node_name: str) -> float:
        """Calculate and return Δv between two nodes in km/s

        Args:
            dep_node_name: name of departure node
            arr_node_name: name of arrival node
        Returns:
            float: Δv between two nodes in km/s
        """
        if set([dep_node_name, arr_node_name]) == set(["Earth", "LEO"]):
            return 0  # because Δv is provided by launch vehicles, not SCs
        if set([dep_node_name, arr_node_name]) == set(["LEO", "LLO"]):
            return 4.04  # km/s
        if set([dep_node_name, arr_node_name]) == set(["LLO", "LS"]):
            return 1.87  # km/s
        # TODO: implement more nodes
        else:
            NotImplementedError(
                f"Δv between {dep_node_name} and {arr_node_name} is not implemented"
            )
            return 0

    def is_holdover_arc(self, dep_node_id: int, arr_node_id: int) -> bool:
        """check if arc is a holdover arc
        Args:
            dep_node_id: departure node id
            arr_node_id: arrival node id
        Returns:
            bool: True if arc is a holdover arc, False otherwise
        """
        return dep_node_id == arr_node_id

    def is_transportation_arc(self, dep_node_id: int, arr_node_id: int) -> bool:
        """check if arc is a transportation arc
        Args:
            dep_node_id: departure node id
            arr_node_id: arrival node id
        Returns:
            bool: True if arc is a transportation arc, False otherwise
        """
        return dep_node_id != arr_node_id

    def is_feasible_arc(self, dep_node_id: int, arr_node_id: int) -> bool:
        """check if arc is feasible

        if arc is a transportation arc, it should be between adjacent nodes
        An holdover arc is not allowed at Earth and LEO nodes

        Args:
            dep_node_id: departure node id
            arr_node_id: arrival node id
        Returns:
            bool: True if arc is feasible, False otherwise
        """
        if not self.node.is_path_graph:
            NotImplementedError("Feasibility not implemented for non-path graphs")
        if self.is_transportation_arc(dep_node_id, arr_node_id):
            return abs(dep_node_id - arr_node_id) == 1
        if self.is_holdover_arc(dep_node_id, arr_node_id):
            node_name = self.node_dict.inverse[dep_node_id]
            return node_name in self.node.holdover_nodes
        return False

    def is_outbound_arc(self, dep_node_id: int, arr_node_id: int) -> bool:
        """check if arc is outbound from Earth node

        Args:
            dep_node_id: departure node id
            arr_node_id: arrival node id
        Returns:
            bool: True if arc is inbound, False otherwise
        """
        # FIX: not necessarily true when more nodes are involved
        return dep_node_id < arr_node_id

    def is_inbound_arc(self, dep_node_id: int, arr_node_id: int) -> bool:
        """check if arc is inbound to Earth node

        Args:
            dep_node_id: departure node id
            arr_node_id: arrival node id
        Returns:
            bool: True if arc is inbound, False otherwise
        """
        # FIX: not necessarily true when more nodes are involved
        return dep_node_id > arr_node_id

    def can_operate_ISRU(self, dep_node_id: int, arr_node_id: int) -> bool:
        """check if ISRU can operate over the given arc

        Args:
            dep_node_id: departure node id
            arr_node_id: arrival node id
        Returns:
            bool: True if ISRU units can operate, False otherwise
        """

        if (
            self.use_isru
            and self.is_holdover_arc(dep_node_id, arr_node_id)
            and dep_node_id == self.node_dict["LS"]
        ):
            return True
        else:
            return False

    def _set_demand(self) -> None:
        """Define demand of each vertex for deterministic case

        It first define entries for deterministic case, then calls
        another method for stochastic case (if necessary)
        to overwrite the array entries.

        Positive quantity indicates supply, negative quantity indicates demand
        Some commodities have infinite supply at Earth node.
        """
        self._set_inf_supply()
        for mis_id, scnr in product(range(self.n_mis), range(self.n_scenarios)):
            mis_start_date_id = self.date_to_time_idx_dict[self.mis_start_dates[mis_id]]
            mis_end_date_id = self.date_to_time_idx_dict[self.mis_end_dates[mis_id]]
            self.int_com_demand[self.node_dict["Earth"]][self.int_com_dict["crew #"]][
                mis_start_date_id
            ][scnr] = self.n_crew
            self.int_com_demand[self.node_dict["LS"]][self.int_com_dict["crew #"]][
                mis_start_date_id
            ][scnr] = -self.n_crew
            self.cnt_com_demand[self.node_dict["LS"]][self.cnt_com_dict["habitat"]][
                mis_start_date_id
            ][scnr] = -self.habit_pl_mass_ls[mis_id]
            self.int_com_demand[self.node_dict["Earth"]][self.int_com_dict["crew #"]][
                mis_end_date_id
            ][scnr] = -self.n_crew
            self.int_com_demand[self.node_dict["LS"]][self.int_com_dict["crew #"]][
                mis_end_date_id
            ][scnr] = self.n_crew
            self.cnt_com_demand[self.node_dict["LS"]][self.cnt_com_dict["consumption"]][
                mis_end_date_id
            ][scnr] = -self.n_crew * self.t_surf_mis * self.mis.net_consumption_cost
            self.cnt_com_demand[self.node_dict["Earth"]][self.cnt_com_dict["sample"]][
                mis_end_date_id
            ][scnr] = -self.sample_mass_ls[mis_id]
        if self.is_stochastic:
            self._set_stochastic_demand()

    def _set_inf_supply(self) -> None:
        """Set infinite supplies for certain commodities.

        Some have unlimited supply at Earth node
        in the begging of the mission.
        Return sample also have unlimited supply at lunar surface node
        in return missions.
        """
        for pl_name, mis_start_date, scnr in product(
            self.comdty.com_names_w_unlim_earth_supply,
            self.mis_start_dates,
            range(self.n_scenarios),
        ):
            if pl_name in self.int_com_names:
                self.int_com_demand[self.node_dict["Earth"]][
                    self.int_com_dict[pl_name]
                ][self.date_to_time_idx_dict[mis_start_date]][scnr] = float("inf")
            elif pl_name in self.cnt_com_names:
                self.cnt_com_demand[self.node_dict["Earth"]][
                    self.cnt_com_dict[pl_name]
                ][self.date_to_time_idx_dict[mis_start_date]][scnr] = float("inf")

        for mis_end_date, scnr in product(
            self.mis_end_dates,
            range(self.n_scenarios),
        ):
            mis_end_date_id = self.date_to_time_idx_dict[mis_end_date]
            self.cnt_com_demand[self.node_dict["LS"]][self.cnt_com_dict["sample"]][
                mis_end_date_id
            ][scnr] = float("inf")

    def _set_stochastic_demand(self) -> None:
        """Overwrite deterministic demand array for stochastic case."""
        for scnr in range(self.n_scenarios):
            for second_mis_start_date in self.second_mis_start_dates:
                self.cnt_com_demand[self.node_dict["LS"]][self.cnt_com_dict["habitat"]][
                    self.date_to_time_idx_dict[second_mis_start_date]
                ][scnr] = -self.habit_pl_mass_2nd[scnr]
            for second_mis_end_date in self.second_mis_end_dates:
                self.cnt_com_demand[self.node_dict["Earth"]][
                    self.cnt_com_dict["sample"]
                ][self.date_to_time_idx_dict[second_mis_end_date]][
                    scnr
                ] = -self.sample_mass_2nd[scnr]

    def _set_final_to_initial_mass_frac_for_arcs(self) -> None:
        """Δv and propellant mass fraction"""
        for i, j, date in product(
            range(self.n_nodes), range(self.n_nodes), self.time_steps
        ):
            i_name, j_name = self.node_dict.inverse[i], self.node_dict.inverse[j]
            date_id = self.date_to_time_idx_dict[date]
            if self.is_feasible_arc(i, j) and self.is_transportation_arc(i, j):
                delta_v_m_s = self._get_delta_v_km_s(i_name, j_name) * 1000
                self.fin_ini_mass_frac[i][j][date_id] = 1 - np.exp(
                    -delta_v_m_s / (self.sc.isp * self.sc.g0)
                )

    def _set_actual_arc_time(self) -> None:
        """actual time of flight for oxygen, food, water"""
        for i, j in product(range(self.n_nodes), range(self.n_nodes)):
            i_name, j_name = self.node_dict.inverse[i], self.node_dict.inverse[j]
            if not self.is_feasible_arc(i, j):
                continue
            if self.is_transportation_arc(i, j):
                self.real_arc_time[i][j] = self._get_time_of_flight(i_name, j_name)
            if self.is_holdover_arc(i, j):
                self.real_arc_time[i][j] = self._get_holdover_time(i_name)

    def get_real_date_from_mis_start(
        self, dep_node: str, arr_node: str, is_outbound: bool
    ) -> int:
        if not self.node.is_path_graph:
            NotImplementedError("Real dates not implemented for non-path graphs")
        dep_node_id = self.node_dict[dep_node]
        arr_node_id = self.node_dict[arr_node]
        assert self.is_feasible_arc(dep_node_id, arr_node_id), """
        Arc from {} to {} is not feasible""".format(dep_node, arr_node)

        if is_outbound:
            return sum(
                self.real_arc_time[node_id][node_id + 1]
                for node_id in range(arr_node_id)
            )
        else:
            return sum(
                self.real_arc_time[node_id][node_id + 1]
                for node_id in range(arr_node_id, self.n_nodes - 1)
            )

    def _set_delta_t(self) -> None:
        """Calculate Δt for mass balance constraints

        Δt is non-zero only for holdover arcs at LLO and LS nodes
        This is because mission is done instataneously (in 1 day)
        in optimization problem for computational purposes.
        t - Δt should refer to the previous time step.
        so, Δt = 1 if t is at mission end date (assuming mission is 1 day)
        and if t is at mission start date, Δt = mission time interval -1
        so that t - Δt is previous mission end date.
        """
        for i, j, date in product(
            range(self.n_nodes), range(self.n_nodes), self.time_steps
        ):
            node_name = self.node_dict.inverse[i]
            if not self.is_holdover_arc(i, j):
                continue
            if node_name not in self.node.holdover_nodes:
                continue
            date_id = self.date_to_time_idx_dict[date]
            if date in self.mis_end_dates:
                self.delta_t[i][j][date_id] = 1
            if date in self.mis_start_dates:
                self.delta_t[i][j][date_id] = self.mis.time_interval - 1

    def _set_allowed_time_window(self) -> None:
        """Specifies allowed time windows.

        A feasible holdover arc are always allowed.
        A feasible outbound transportation arc is allowed
        only for outbound mission dates,
        and a feasible inbound transportation arc is allowed
        only for return mission dates.
        """
        for i, j, n in product(
            range(self.n_nodes), range(self.n_nodes), range(self.n_mis)
        ):
            if not self.is_feasible_arc(i, j):
                continue
            if self.is_transportation_arc(i, j):
                self.allowed_time_window[i][j].append(
                    self.is_inbound_arc(i, j) * 1 + n * self.mis.time_interval
                )
            if self.is_holdover_arc(i, j):
                self.allowed_time_window[i][j].extend(
                    [n * self.mis.time_interval, n * self.mis.time_interval + 1]
                )

    def _set_ISRU_work_time(self) -> None:
        """Sets work time for ISRU plant at LS node.

        ISRU work time is time between the previous and current time step.
        """
        holdover_win_on_LS: list = self.allowed_time_window[self.node_dict["LS"]][
            self.node_dict["LS"]
        ]
        for date_id in range(len(holdover_win_on_LS) - 1):
            self.isru_work_time[self.node_dict["LS"]][date_id] = (
                holdover_win_on_LS[date_id + 1] - holdover_win_on_LS[date_id]
            )

    def _define_date_to_time_idx_dict(self) -> None:
        """Defines dict to convert date to time index and vice versa.

        e.g., if time_steps = [0, 1, 365, 366],
        date_to_time_idx_dict[365] returns 2
        """
        date_to_time_idx_dict: dict[int, int] = {}
        for date in self.time_steps:
            date_to_time_idx_dict[date] = self.time_steps.index(date)
        self.date_to_time_idx_dict = bidict(date_to_time_idx_dict)
