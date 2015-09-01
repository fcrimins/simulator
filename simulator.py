import sys
import copy
import numpy as np
import pyodbc
import collections
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
from restriction import Restriction
from tcost import TCostCalculator
from opt import PortfolioOptimizer


# global simulator configuration dictionary
config = OpenStruct()

class Simulator(object):

    _float_t = np.float64

    def __init__(self, config_name=None):
        """Initialize a simulation.
        @param config_name: Configuration to checkout from the database or None to select one at random
            and wait until there is one if none is ready to be checked out.
        """
        global config
        config.load(config_name)

        # the OMS contains all of the state, e.g. portfolio + residual/decayed slippage
        self._oms = SimOMS()

    def run(self):
        """Runs a simulation from start of ramp up until (exclusive) end date."""
        global config
        for dt in config.calendar:
            self._bod(dt)
            self._simulate_date(dt)
            self._eod(dt)  # process end-of-day

    def _simulate_date(self, dt):
        """Simulate a single date.
        @param dt: The date to simulate.
        """
        # @TODO: intraday optimizations should be considered part of the order slicer/scheduler (in prod,
        # at least), which should be responsible for satisfying breached constraints, which currently
        # happens in the optimizer, which is the proper place assuming the first statement of this comment
        if config.isintraday:
            for t, next_t in config.calendar.day(dt):
                orders = self._rebalance(dt, (t, next_t))
                self._oms.send(orders)
        else:
            orders = self._rebalance(dt)
            self._oms.send(orders)

    def _bod(self, dt):
        self._oms.bod(dt)  # apply splits and spinoffs
        # @TODO: initialize the optimizer for the day, i.e. pull stuff out of the OMS and other locations
        # should the simulator just get its prices from the OMS, i.e. should prices be "owned" by the
        # market and should the simulator interface with everything owned by the market through the OMS?
        # this would consolidate the interface to prices at the very least

        optin = {}

        # instruments
        ninsts = 2
        optin['resid_vars'] = np.array([0.22, 0.17], dtype=self._float_t)**2 / config.days_per_year
        optin['bod_positions'] = np.array([i.bod_shares for i in self._oms._state], dtype=self._float_t)
        optin['hmaxs'] = np.array([config.hard_max_position_value] * ninsts, dtype=self._float_t)
        optin['hmins'] = -optin['hmaxs']
        optin['smaxs'] = np.array([config.soft_max_position_value] * ninsts, dtype=self._float_t)
        optin['smins'] = -optin['smaxs']
        optin['mdvs'] = np.array([1e6, 1e5], dtype=self._float_t)  # shares units (because "volumes" are shares)
        optin['round_lots'] = np.array([100, 100], dtype=self._float_t)

        # risk model
        nfactors = 2
        optin['factor_vars'] = np.array([0.02, 0.09])**2 / config.days_per_year
        optin['factor_corrs'] = np.ones((nfactors, nfactors), dtype=self._float_t)
        optin['factor_corrs'][0, 1] = 0.2
        optin['factor_corrs'][1, 0] = 0.2

        optin['loadings'] = [list()] * ninsts
        optin['loadings'][0] = [(0, 1.1), (1, 0.5)]
        optin['loadings'][1] = [(0, 0.9), (1, 1.5)]

        # parameters
        optin['horizon'] = config.horizon
        optin['mvk'] = config.mvk
        optin['convergence_threshold'] = config.convergence_threshold
        optin['mdv_position_limit'] = config.mdv_position_limit
        optin['mdv_trading_limit'] = config.mdv_trading_limit

        optin['maxgmv'] = config.maxgmv
        optin['maxgmv_buffer'] = config.maxgmv_buffer
        optin['maxgmv_penaltyrate'] = config.maxgmv_penaltyrate

        # initialize optimizer
        self._optimizer = PortfolioOptimizer(**optin)

    def _eod(self, dt):
        self._oms.eod(dt)

    def _rebalance(self, dt, ts=None):
        """Feel free to override this implementation."""

        # set up constrained optimization limits (if intraday)
        if ts is not None:
            t, next_t = ts  # in seconds
            minutes = (next_t - t) / 60.0
            scaled_per_opt_lim = config.per_optimization_mdv_trading_limit * minutes
            self._optimizer._per_optimization_mdv_trading_limit = scaled_per_opt_lim

        # @TODO: construct istate
        istate = None

        fstate = self._optimizer.optimize(istate)

        orders = np.zeros(len(istate), dtype=self._float_t)
        for i, (ist, fst) in enumerate(zip(istate, fstate)):
            orders[i] = fst.shares - ist.shares
        return orders


class SimOMS(object):

    def __init__(self):
        ninsts = 4000  # @TODO
        self._fill_model = self.FillModel()
        self._tcost_calculator = TCostCalculator()
        self._state = [self.OMSInst(0.0, 0.0, 0.0, Restriction.NONE) for _ in range(ninsts)]
        self._slip_per_shr = [0.0] * ninsts  # stored outside of _state b/c it only applies to the simulator, not prod

    def bod(self, dt):
        self._apply_cax(dt)
        self._decay_slippage()

    def _apply_cax(self, dt):
        # @TODO: apply splits and spinoffs--real spinoffs, from DataStream!  and set bod_shares
        # apply splits and spinoffs to both positions and slippage_per_share
        pass

    def _decay_slippage(self):
        """Could make this decay intraday eventually.  For now, assume 1 day of volume."""
        global config
        self._bod_slip_per_shr = copy.copy(self._slip_per_shr)
        for i in range(len(self._slip_per_shr)):
            # @TODO: should daily_slippage_decay_rate = sqrt(slippage model exponent)
            self._slip_per_shr[i] *= config.daily_slippage_decay_rate

    def eod(self, dt):
        """Aggregate trades and output position state files and daily simulator stats."""
        ninsts = len(self._state)
        eod_state = np.zeros((ninsts, 2), dtype=Simulator._float_t)  # EOD position and slip_per_shr
        daily_pnl = self.DailyPnLStats()

        for i, (inst, eodslip, bodslip) in zip(self._state, self._slip_per_shr, self._bod_slip_per_shr):
            daily_pnl.update(inst, eodslip, bodslip)
            eod_state[i, 0] = inst.shares
            eod_state[i, 1] = eodslip

    class DailyPnLStats(object):
        def __init__(self):
            global config
            self._tcost_calculator = config.tcost_calculator
            self.residual_pnl = 0.0  # resid and factor are based on raw, non-slippage-adjusted prices
            self.factor_pnl = 0.0
            self.holding_slip_decay_pnl = 0.0  # pnl due to bod position based on change in slip/shr...due to decay
            self.holding_intrdy_slip_pnl = 0.0  # ...due to intraday changes in aggregate slippage per share
            self.delta_px_trading_pnl = 0.0  # trading pnl due to raw changes in price
            self.trading_slip_pnl = 0.0  # slippage pnl due to shares traded today
            self.commissions_pnl = 0.0
            self.holding_costs_pnl = 0.0
            self.value_traded = 0.0

        @property
        def holding_pnl(self):
            """Includes dividend PnL.  One could argue that holding_slip_decay_pnl belongs here."""
            return self.residual_pnl + self.factor_pnl + self.holding_costs_pnl

        @property
        def trading_pnl(self):
            """PnL caused by trading, which is slightly different from PnL of trades."""
            return (self.holding_slip_decay_pnl + self.holding_intrdy_slip_pnl +
                    self.delta_px_trading_pnl + self.trading_slip_pnl + self.commissions_pnl)

        def update(self, inst, eodslip_pshr, bodslip_pshr):
            """Update daily PnL components based on a single instrument's positions and trades."""
            # can't confirm that total computed marginal slippage is equal to individual trade aggregated
            # slippage (in intraday sims, at least) b/c the latter is computed using intraday prices
            decay_rate = config.daily_slippage_decay_rate
            if __debug__:
                totshrs = sum(s for s, _ in inst.trades)
                totslip = self._tcost_calculator.slippage(totshrs, inst.price, inst.mdv)
                totslip_pshr = totslip / totshrs
                assert(eodslip_pshr == bodslip_pshr * decay_rate + totslip_pshr)  # only true in daily sims

            daily_return = inst.excess_return + inst.expected_return
            raw_eod_px = inst.price
            raw_bod_px = raw_eod_px / (1.0 + daily_return)  # really yesterday's EOD price adjusted for cax

            if __debug__:
                _adj_eod_px = raw_eod_px + eodslip_pshr
                _adj_bod_post_decay_px = raw_bod_px + bodslip_pshr * decay_rate
                _adj_bod_px = raw_bod_px + bodslip_pshr

            # note that these include dividend PnL
            self.residual_pnl += inst.bod_shares * raw_bod_px * inst.excess_return
            self.factor_pnl   += inst.bod_shares * raw_bod_px * inst.expected_return

            # PnL due to change in price due to daily slippage decay
            self.holding_slip_decay_pnl += inst.bod_shares * (bodslip_pshr * decay_rate - bodslip_pshr)

            # PnL due to change in price due to new intraday slippage
            self.holding_intrdy_slip_pnl += inst.bod_shares * (eodslip_pshr - bodslip_pshr * decay_rate)

            # subtract to convert "cost" to "PnL"
            self.holding_costs_pnl -= self._tcost_calculator._holding_costs(inst.bod_shares, inst.price, 1)

            for t in inst.trades:
                # raw_eod_px + eodslip_pshr - (t.price + t.slip_pshr)
                self.delta_px_trading_pnl += t.shares * (raw_eod_px - t.price)  # due to raw price changes
                self.trading_slip_pnl += t.shares * (eodslip_pshr - t.slip_pshr)  # due to slippage changes
                self.commissions_pnl -= self._tcost_calculator._commissions(t.shares)
                self.value_traded += np.fabs(t.shares * (t.price + t.slip_pshr))

    def send(self, orders):
        """Send the list of orders to the (simulated) market.
        @param orders: List of orders (in shares units) one per every instrument regardless of 0.
        """
        for i, (ord, inst) in enumerate(zip(orders, self._state)):
            if ord == 0:
                continue
            posmv = inst.shares * inst.price
            ordmv = ord * inst.price
            if Restriction.constraints_satisfied(inst.restriction, posmv, ordmv):

                # apply fill ratio (this is one component of the simulated market, i.e. fill shares)
                fillshrs = self._fill_model.ratio() * ord

                # update aggregate slippage (another component, i.e. fill price)
                ipos = inst.bod_shares
                fpos = inst.shares
                mslip = self._tcost_calculator.marginal_slippage(ipos, fpos, fillshrs, inst.price, inst.mdv)
                mslip_per_shr = mslip / fillshrs
                self._slip_per_shr[i] += mslip_per_shr

                # record fill shares and (the 2 components of) fill price
                inst.record_trade(fillshrs, inst.price, self._slip_per_shr[i])

    class OMSInst(object):
        """Instrument class to hold the state of an individual instrument's data in the OMS.
        This class should remain independent of whether we're simulating or production trading.
        @param shares: The position in the instrument (in shares, obviously).
        @param bod_shares: The position at the beginning of the day.
        @param restriction: The trading restriction on the instrument (for double checking the optimizer).
        @param trades: List of (shares, price, slip/shr) tuples representing trades.
        """
        __slots__ = ['shares', 'bod_shares', 'restriction', 'trades']
        def __init__(self, *args):
            for i, v in enumerate(args):
                setattr(self, self.__slots__[i], v)
            self.trades = []

        @property
        def price(self):
            return np.nan  # @TODO

        @property
        def mdv(self):
            """Median daily volume (in shares, b/c volume should always be in shares)."""
            return np.nan  # @TODO

        @property
        def excess_return(self):
            return np.nan  # @TODO, should calling this method trigger the setting of these values for all insts?

        @property
        def expected_return(self):
            return np.nan  # @TODO

        Trade = collections.namedtuple('Trade', 'shares price slip_pshr'.split(' '))
        def record_trade(self, fillshrs, fillpx, slip_pshr):
            self.shares += fillshrs
            self.trades.append(self.Trade(fillshrs, fillpx, slip_pshr))

    class FillModel(object):
        """Obviously this should be more complex and depend on intraday data."""
        def ratio(self, *args):
            """Return the fill ratio for some as yet unknown set of inputs."""
            return 0.9


if __name__ == "__main__":
    
    if len(sys.argv) > 2:
        raise ValueError('sys.argv ({}) length > 2'.format(sys.argv))
    
    config_name = sys.argv[1] if len(sys.argv) == 2 else None
    sim = Simulator(config_name=config_name)
    
    sim.run()
    
    # don't output perhaps? just let some other program come along and do it
    # or, yes, output, but also make it easy for another program to do so, e.g.
    # by constructing its own Simulator instance and reading stats from the log file
    # sim.out()