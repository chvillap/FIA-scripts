"""Python implementation of simple agents for AI.

To set up the virtual environment:
    conda create -n agents jupyter
    source activate agents
    pip install grid
"""

import collections
import random
import copy

from statistics import mean

# -----------------------------------------------------------------------------


class Entity(object):
    """Represents any named physical object that can appear in an environment.
    """

    def __repr__(self):
        name = getattr(self, '__name__', self.__class__.__name__)
        return '<{}>'.format(name)

    def is_alive(self):
        """Tells whether the entity is alive or not.
        """
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Show the entity's internal state. Must be overriden by subclasses.
        """
        raise NotImplementedError('Subclasses must implement abstract methods')

    def display(self, canvas, x, y, width, height):
        """Displays an image of the entity. Must be overriden by subclasses.
        """
        raise NotImplementedError('Subclasses must implement abstract methods')

# -----------------------------------------------------------------------------


class Agent(Entity):
    """An abstract agent, which is also an entity. It receives perceptions and
    return actions in response. Optionally, it can also use a performance
    measure to evaluate how well it performs.

    Notice that the agent's program is an attribute, not a method. And the
    exact definition of what is a perception and what is an action depends on
    a particular environment.
    """

    def __init__(self, program=None):
        self.alive = True
        self.bump = False
        self.holding = []
        self.performance = 0

        if program is None:
            def program(perception):
                return eval('perception={}; action? '.format(perception))
        assert isinstance(program, collections.Callable)
        self.program = program

    def can_grab(self):
        """Tells whether this agent can grab objects. Must be overriden by subclasses.
        """
        raise NotImplementedError('Subclasses must implement abstract methods')

# -----------------------------------------------------------------------------


def TableDrivenAgentProgram(table):
    """Defines a program that selects an action based on the current sequence
    of acquired perceptions. The table must be a dict with entries in the form:
    {perception_sequence: action}
    """
    perceptions = []

    def program(perception):
        perceptions.append(perception)
        action = table.get(tuple(perceptions))
        return action

    return program


def RandomAgentProgram(actions):
    """Defines a program that picks actions randomly, ignoring any perception.
    """
    return lambda perception: random.choice(actions)


def SimpleReflexAgentProgram(rules, interpret_input):
    """Defines a program that uses a given function to interpret the perception
    and get its current state, and then uses a set of given rules to produce a
    proper action in response.
    """
    def program(perception):
        state = interpret_input(perception)
        rule = rule_match(state, rules)
        action = rule.action
        return action

    return program


def ModelBasedReflexAgentProgram(rules, update_state):
    """Defines a program that uses a given function to update its internal
    state based on its current state, action and perception, and then uses a
    set of given rules to produce a proper action in response.
    """
    def program(perception):
        program.state = update_state(program.state, program.action, perception)
        rule = rule_match(program.state, rules)
        action = rule.action
        return action

    return program


def rule_match(state, rules):
    """Finds the first rule that matches the given state.
    """
    for rule in rules:
        if rule.matches(state):
            return rule

# -----------------------------------------------------------------------------


class Environment(object):
    """An abstract environment. All subclasses must implement the methods:
        give_perception(): specify an agent's perception in the environment.
        execute_action(): updates the environment based on a performed action.

    The environment holds a list of entities and agents. Every agent has a
    performance measure which will be updated after each action. And every
    entity has a location, even though some environments don't need this.
    """

    def __init__(self):
        self.entities = []
        self.agents = []

    def give_perception(self, agent):
        """Returns the perception an agent gets from the environment.
        Must be overriden by subclasses.
        """
        raise NotImplementedError('Subclasses must implement abstract methods')

    def execute_action(self, agent, action):
        """Updates the environment based on some action.
        Must be overriden by subclasses.
        """
        raise NotImplementedError('Subclasses must implement abstract methods')

    def entity_classes(self):
        """Returns a list of the classes of all entities in this environment.
        Must be overriden by subclasses.
        """
        raise NotImplementedError('Subclasses must implement abstract methods')

    def default_location(self, entity):
        """Default location for putting some new entity.
        Must be overriden by subclasses.
        """
        raise NotImplementedError('Subclasses must implement abstract methods')

    def exogenous_change(self):
        """Executes spontaneous changes in the environtment.
        Must be overriden by subclasses.
        """
        raise NotImplementedError('Subclasses must implement abstract methods')

    def is_done(self):
        """Tells whether there aren't any agents alive in the environment
        anymore (stop condition).
        """
        return not any(agent.is_alive() for agent in self.agents)

    def step(self):
        """Executes a single step for all agents in the environment.
        """
        if not self.is_done():
            actions = []
            for agent in self.agents:
                if agent.alive:
                    perception = self.give_perception(agent)
                    action = agent.program(perception)
                    actions.append(action)
                else:
                    actions.append('')

            for agent, action in zip(self.agents, actions):
                self.execute_action(agent, action)
            self.exogenous_change()

    def run(self, steps=1000):
        """Executes a number of step interactions between the environment and
        its agents.
        """
        for step in range(steps):
            if self.is_done():
                return
            self.step()

    def list_entities_at(self, location, eclass=Entity):
        """Returns all entities of some specific class at some given location.
        """
        return [entity for entity in self.entities
                if entity.location == location and isinstance(entity, eclass)]

    def some_entities_at(self, location, eclass=Entity):
        """Tells whether there is at least one entity of some specific class
        at some given location.
        """
        return self.list_entities_at(location, eclass) != []

    def add_entity(self, entity, location=None):
        """Adds an entity to the environment. If this entity is a program, an
        agent is built to hold it.
        """
        if not isinstance(entity, Entity):
            entity = Agent(entity)

        assert entity not in self.entities, "Can't add the same entity twice"

        if location is None:
            entity.location = self.default_location(entity)
        else:
            entity.location = location

        self.entities.append(entity)
        if isinstance(entity, Agent):
            entity.performance = 0
            self.agents.append(entity)

    def delete_entity(self, entity):
        """Removes a particular entity from the environment.
        """
        try:
            self.entities.remove(entity)
        except ValueError as e:
            print(e)
            print('  in Environment delete_entity')
            print('  Entity to be removed: {} at {}'.format(
                entity, entity.location))
            print('  from list: {}'.format(
                [(entity, entity.location) for entity in self.entities]))

        if entity in self.agents:
            self.agents.remove(entity)


class Obstacle(Entity):
    """Something that can prevent an agent to move to some location in an
    environment."""
    pass


class Wall(Obstacle):
    """A particular kind of obstacle."""
    pass

# -----------------------------------------------------------------------------


class Direction:
    """Represents a direction for agents that move on a 2D plane.
    """

    L = 'left'
    U = 'up'
    R = 'right'
    D = 'down'

    def __init__(self, direction):
        self.direction = direction

    def __add__(self, heading):
        if self.direction == self.R:
            sides = {self.R: Direction(self.D),
                     self.L: Direction(self.U)}
            return sides.get(heading, None)
        elif self.direction == self.L:
            sides = {self.R: Direction(self.U),
                     self.L: Direction(self.L)}
            return sides.get(heading, None)
        elif self.direction == self.U:
            sides = {self.R: Direction(self.R),
                     self.L: Direction(self.L)}
            return sides.get(heading, None)
        elif self.direction == self.D:
            sides = {self.R: Direction(self.L),
                     self.L: Direction(self.R)}
            return sides.get(heading, None)

    def move_forward(self, from_location):
        """Moves one position forward from some given location.
        """
        x, y = from_location
        if self.direction == self.R:
            return (x + 1, y)
        elif self.direction == self.L:
            return (x - 1, y)
        elif self.direction == self.U:
            return (x, y - 1)
        elif self.direction == self.D:
            return (x, y + 1)

# -----------------------------------------------------------------------------


def trace_agent(agent):
    """Traces an agent by replacing its program by another one that prints the
    agent's perception and action after executing. This allows us to see what
    the agent is doing in the environment.
    """
    old_program = agent.program

    def new_program(perception):
        action = old_program(perception)
        print('{} percepts {} and does {}'.format(agent, perception, action))
        return action

    agent.program = new_program
    return agent


def compare_agents(EnvironmentFactory, AgentFactories, n=10, steps=1000):
    """Returns the mean performance scores obtained by all agents in some
    environment.
    """
    environments = [EnvironmentFactory() for i in range(n)]
    return [(A, test_agent(A, steps, copy.deepcopy(environments)))
            for A in AgentFactories]


def test_agent(AgentFactory, steps, environments):
    """Computes the mean of the performance scores obtained by an agent in
    different environments.
    """
    def score(environment):
        agent = AgentFactory()
        environment.add_entity(agent)
        environment.run(steps)
        return agent.performance

    return mean(map(score, environments))
