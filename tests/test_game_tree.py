"""Structure and payoff tests for the river game tree."""

from game_tree import IP, OOP, Decision, Terminal, build_tree, decision_nodes

POT = 100.0


def make_tree():
    return build_tree(pot=POT, bet_fractions=(0.5, 1.0), raise_fractions=(1.0,), max_raises=1)


def test_root_is_oop_with_check_and_bets():
    root = make_tree()
    assert root.player == OOP
    assert root.actions == ("check", "bet 50", "bet 100")


def test_check_check_is_showdown():
    root = make_tree()
    ip_node = root.children["check"]
    assert ip_node.player == IP
    showdown = ip_node.children["check"]
    assert isinstance(showdown, Terminal)
    assert showdown.folder is None
    assert showdown.invested == (0.0, 0.0)


def test_facing_bet_has_fold_call_raise():
    root = make_tree()
    ip_vs_bet = root.children["bet 100"]
    assert ip_vs_bet.player == IP
    assert ip_vs_bet.actions == ("fold", "call", "raise 100")


def test_raise_is_capped():
    # After one raise, the original bettor may only fold or call.
    root = make_tree()
    oop_vs_raise = root.children["bet 100"].children["raise 100"]
    assert oop_vs_raise.player == OOP
    assert oop_vs_raise.actions == ("fold", "call")


def test_bet_and_raise_amounts():
    root = make_tree()
    # OOP bets 100% of the 100 pot.
    ip_vs_bet = root.children["bet 100"]
    # IP pot-raises: call 100, then raise by pot-after-call (100+100+100=300).
    oop_vs_raise = ip_vs_bet.children["raise 100"]
    call_terminal = oop_vs_raise.children["call"]
    assert call_terminal.invested == (400.0, 400.0)


def test_fold_payoffs():
    root = make_tree()
    # OOP bets 50 into 100, IP folds: OOP wins the pot, IP loses nothing
    # on the river (their fold terminal investment is 0).
    fold = root.children["bet 50"].children["fold"]
    assert fold.payoffs(showdown_winner=None) == (100.0, 0.0)

    # OOP bets 100, IP raises pot, OOP folds: IP wins pot + OOP's 100.
    fold = root.children["bet 100"].children["raise 100"].children["fold"]
    assert fold.payoffs(showdown_winner=None) == (-100.0, 200.0)


def test_showdown_payoffs():
    root = make_tree()
    showdown = root.children["bet 100"].children["call"]
    assert showdown.invested == (100.0, 100.0)
    assert showdown.payoffs(showdown_winner=OOP) == (200.0, -100.0)
    assert showdown.payoffs(showdown_winner=IP) == (-100.0, 200.0)
    assert showdown.payoffs(showdown_winner=None) == (50.0, 50.0)


def test_all_payoffs_sum_to_pot():
    # Constant-sum check across every terminal and outcome.
    def walk(node):
        if isinstance(node, Terminal):
            for winner in (OOP, IP, None):
                payoff = node.payoffs(winner)
                assert abs(sum(payoff) - POT) < 1e-9
            return
        for child in node.children.values():
            walk(child)

    walk(make_tree())


def test_decision_node_count():
    # root, IP-after-check, 2 OOP-vs-bet nodes, 2 IP-vs-raise nodes,
    # 2 IP-vs-bet nodes, 2 OOP-vs-raise nodes = 10 decision points.
    assert len(decision_nodes(make_tree())) == 10
