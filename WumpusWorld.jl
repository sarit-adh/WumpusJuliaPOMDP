
using POMDPs
import Base: ==


#Define world state which includes agent location, gold location, wumpus location and done flag
type WorldState
    agentLoc_x::Int64
    agentLoc_y::Int64
    goldLoc_x::Int64
    goldLoc_y::Int64
    wumpusLoc_x::Int64
    wumpusLoc_y::Int64
    done::Bool
end

#Immutables are compared by the value of their fields, whereas mutables
#by ===, i.e. whether they are same object in memory.
#But can overload ==
==(s1::WorldState, s2::WorldState) = s1.agentLoc_x == s2.agentLoc_x && s1.goldLoc_x == s2.goldLoc_x && s1.agentLoc_y == s2.agentLoc_y && s1.goldLoc_y == s2.goldLoc_y && s1.wumpusLoc_x==s2.wumpusLoc_x && s1.wumpusLoc_y==s2.wumpusLoc_y && s1.done==s2.done

#posequal(s1::WorldState, s2::WorldState) = s1.agentLoc == s2.agentLoc && s1.goldLoc == s2.goldLoc


#Wumpus POMDP inherits from POMDP and includes parameter for setting up the grid world
type WumpusPOMDP <: POMDP{WorldState, Symbol, Symbol}
    size_x::Int64
    size_y::Int64
    r_gold::Float64
    r_wumpus::Float64
    discount_factor::Float64
end

#CHANGE wumpus POMDP parameters here
function WumpusPOMDP()
    return WumpusPOMDP(3,3,1000,-1000,0.99) #(size_x,size_y,r_gold,r_wumpus,discount_factor)
end

#function to check if agent is in same square as gold
function colocate_gold(state::WorldState)
      return state.agentLoc_x==state.goldLoc_x && state.agentLoc_y==state.goldLoc_y
end

#function to check if agent is in same square as wumpus
function colocate_wumpus(state::WorldState)
    return state.agentLoc_x==state.wumpusLoc_x && state.agentLoc_y==state.wumpusLoc_y
end

#function to check if agent is in adjacent square to wumpus
function adjacent_wumpus(state::WorldState)
    if ((state.agentLoc_x == state.wumpusLoc_x-1 && state.agentLoc_y == state.wumpusLoc_y) || (state.agentLoc_x == state.wumpusLoc_x+1 && state.agentLoc_y == state.wumpusLoc_y) || (state.agentLoc_x == state.wumpusLoc_x && state.agentLoc_y == state.wumpusLoc_y-1) || (state.agentLoc_x == state.wumpusLoc_x && state.agentLoc_y == state.wumpusLoc_y+1))
        return true
    else
        return false
    end
end
 
#iterate over every possible position of agent, gold and wumpus as well as done value to get list of all states
function POMDPs.states(pomdp::WumpusPOMDP)
    s = WorldState[]
    for agentLoc_x=1:pomdp.size_x,goldLoc_x=1:pomdp.size_x,agentLoc_y=1:pomdp.size_y,goldLoc_y=1:pomdp.size_y,wumpusLoc_x=1:pomdp.size_x,wumpusLoc_y=1:pomdp.size_y,done=0:1
        push!(s,WorldState(agentLoc_x,agentLoc_y,goldLoc_x,goldLoc_y,wumpusLoc_x,wumpusLoc_y,done))
    end
    return s
end

#We have 5 actions in total, since noop is not included, agent is expected to perform grab if no other optimal action is present
POMDPs.actions(::WumpusPOMDP) = [:left,:right,:up,:down,:grab]

POMDPs.actions(pomdp::WumpusPOMDP,state::WorldState) = POMDPs.actions(pomdp)

#Utility function to return index for action symbol
function POMDPs.action_index(::WumpusPOMDP,a::Symbol)
    if a==:left
        return 1
    elseif a==:right
        return 2
    elseif a==:up
        return 3
    elseif a==:down
        return 4
    elseif a==:grab
        return 5
    end
    error("invalid WumpusPOMDP action: $a")
end;

POMDPs.observations(::WumpusPOMDP) = [:stench,:glitter,:stenchandglitter,:none];
POMDPs.observations(pomdp::WumpusPOMDP,s::WorldState) = POMDPs.observations(pomdp)

function POMDPs.obs_index(::WumpusPOMDP,o::Symbol)
    if o==:stench
        return 1
    elseif o==:glitter
        return 2
    elseif o==:stenchandglitter
        return 3
    elseif o==:none
        return 4
    end
    error("invalid WumpusPOMDP action: $a")
end;

#Utility function to normalize probabilites and make sure they sum to 1
function m_normalize!(probabilityVector::Vector{Float64})
    probabilityVector[:] =  probabilityVector / sum(probabilityVector)
end

#structure holding world state and corresponding probability
type WorldStateDistribution
    probabilityVector::Vector{Float64}
    stateVector::Vector{WorldState}
end
function WorldStateDistribution(pomdp::WumpusPOMDP)
    stateVector = POMDPs.states(pomdp)
    probabilityVector = [1. / length(stateVector) for i in 1:length(stateVector) ]
    return WorldStateDistribution(probabilityVector,stateVector)
end

function WorldStateDistribution(pomdp::WumpusPOMDP, agentLoc_x::Int64,agentLoc_y::Int64,done::Bool)
    stateVector = POMDPs.states(pomdp)
    agentProb = 1. / (pomdp.size_x*pomdp.size_y)
    probabilityVector = vec(fill(0.0, 1, length(stateVector)))
    for i in 1:length(stateVector)
       
       if(stateVector[i].done==done && stateVector[i].agentLoc_x == agentLoc_x && stateVector[i].agentLoc_y == agentLoc_y)
            probabilityVector[i] = agentProb
        end
    end
    m_normalize!(probabilityVector)
    if sum(probabilityVector)==0 print("world state distribution 2 sum zero") end
    return WorldStateDistribution(probabilityVector,stateVector)
end

function WorldStateDistribution(pomdp::WumpusPOMDP, state::WorldState)
    stateVector = POMDPs.states(pomdp)
    found = false
    probabilityVector = vec(fill(0.0, 1, length(stateVector)))
    for i in 1:length(stateVector)
       if(stateVector[i]==state)
            probabilityVector[i] = 1
            found = true
        end
    end
    if(!found) print("not found", state) end
    #m_normalize!(probabilityVector)
        if sum(probabilityVector)==0 print("world state distribution 3 sum zero") end
    return WorldStateDistribution(probabilityVector,stateVector)
end

POMDPs.iterator(d::WorldStateDistribution) = d.stateVector

function POMDPs.pdf(d::WorldStateDistribution,state::WorldState)
    return d.probabilityVector[findfirst(d.stateVector, state)]
end

using StatsBase
function POMDPs.rand(rng::AbstractRNG, d:: WorldStateDistribution)
    return sample(d.stateVector, WeightVec(d.probabilityVector))
end

function inbounds(pomdp::WumpusPOMDP,agentLoc_x,agentLoc_y)
    if 1<=agentLoc_x<=pomdp.size_x && 1<=agentLoc_y<=pomdp.size_y
        return true
    else
        return false
    end
end

inbounds(pomdp::WumpusPOMDP,state::WorldState) = inbounds(pomdp,state.agentLoc_x,state.agentLoc_y);

#transition function
function POMDPs.transition(p::WumpusPOMDP, state::WorldState, action::Symbol)
    #println("Transition")
    x = state.agentLoc_x
    y = state.agentLoc_y
    a = POMDPs.action_index(p,action)
    if state.done
         return WorldStateDistribution(p,state)
     end
    
      if action==:grab && colocate_gold(state)
         tempstate = WorldState(x,y,state.goldLoc_x,state.goldLoc_y,state.wumpusLoc_x,state.wumpusLoc_y,true)
         return WorldStateDistribution(p,tempstate)
      end
     if colocate_wumpus(state)
        tempstate = WorldState(x,y,state.goldLoc_x,state.goldLoc_y,state.wumpusLoc_x,state.wumpusLoc_y,true)
        return WorldStateDistribution(p,tempstate)
     end
    
    
    #The neighbor array represents the possible states to which the
    #agent in its current state may transition. The states correspond to 
    #the integer representation of each action.
    neighbor = [
        WorldState(x-1,y,state.goldLoc_x,state.goldLoc_y,state.wumpusLoc_x,state.wumpusLoc_y,state.done),  #left
        WorldState(x+1,y,state.goldLoc_x,state.goldLoc_y,state.wumpusLoc_x,state.wumpusLoc_y,state.done),  #right
        WorldState(x,y+1,state.goldLoc_x,state.goldLoc_y,state.wumpusLoc_x,state.wumpusLoc_y,state.done),  #up
        WorldState(x,y-1,state.goldLoc_x,state.goldLoc_y,state.wumpusLoc_x,state.wumpusLoc_y,state.done),   #down
        WorldState(x,y,state.goldLoc_x,state.goldLoc_y,state.wumpusLoc_x,state.wumpusLoc_y,state.done)      #original cell
    ]
    
    #The target cell is the location at the index of the appointed action.
    target = neighbor[a]
    
    
    #If the target cell is out of bounds, the agent remains in 
    #the same cell. Otherwise the agent transitions to the target 
    #cell.
    if !inbounds(p,target)
        #return SparseCat([state], [1.0])
        
        return WorldStateDistribution(p,state)
    else
        #return SparseCat([target], [1.0])
        
        return WorldStateDistribution(p,target)
    end
end

testState = WorldState(1,1,1,2,1,1,false)

d = POMDPs.transition(WumpusPOMDP(),testState,:right)
d.probabilityVector[findfirst(d.stateVector, WorldState(2,1,2,1,2,1,false))]

#reward function
function POMDPs.reward(pomdp::WumpusPOMDP, state::WorldState, action::Symbol)
    totalReward = 0.0
    if colocate_wumpus(state)
        totalReward+=pomdp.r_wumpus
    elseif action==:grab && colocate_gold(state)
        totalReward+=pomdp.r_gold
    else
        totalReward-=1
    end
    return totalReward
end

POMDPs.reward(WumpusPOMDP(),WorldState(2, 1, 2, 1, 2, 1, false),:right)

type ObservationDistribution
    probabilityVector::Vector{Float64}
    observationVector::Vector{Symbol}
end

function ObservationDistribution(pomdp::WumpusPOMDP)
    observationVector = POMDPs.observations(pomdp)
    total_stenchy_cells = ((pomdp.size_x * pomdp.size_y) - 1)     
    num_possible_middle_pit = (pomdp.size_x - 2) * (pomdp.size_y -2)
    num_possible_corner_pit = 4 #always 4 , no matter size of grid
    num_possible_edge_pit = 2*((pomdp.size_x - 2) + (pomdp.size_y -2))
    cells_stenchy_from_edge_pit = 3
    cells_stenchy_from_corner_pit = 2
    cells_stenchy_from_middle_pit = 4
    prob_stench_from_edge_pit = cells_stenchy_from_edge_pit / total_stenchy_cells
    prob_stench_from_corner_pit = cells_stenchy_from_corner_pit / total_stenchy_cells
    prob_stench_from_middle_pit = cells_stenchy_from_middle_pit / total_stenchy_cells
    prob_stench_only = ((prob_stench_from_edge_pit * num_possible_edge_pit) + (prob_stench_from_corner_pit * num_possible_corner_pit) + (prob_stench_from_middle_pit * num_possible_middle_pit))/ (num_possible_middle_pit+num_possible_corner_pit+num_possible_edge_pit)
    prob_glitter_only = 1/(pomdp.size_x*pomdp.size_y)
    prob_glitter_stench = prob_glitter_only*prob_stench_only
    prob_none = 1 - prob_glitter_only - prob_stench_only - prob_glitter_stench
        
    probabilityVector = [prob_stench_only,prob_glitter_only,prob_glitter_stench,prob_none]
    if sum(probabilityVector)==0 print("observation distribution sum zero") end
    return ObservationDistribution(probabilityVector,observationVector)
    #ObservationDistribution((1/(pomdp.size_x*pomdp.size_y)),[true,false])
end

    
print(ObservationDistribution(WumpusPOMDP()))    

#observation function
function POMDPs.observation(pomdp::WumpusPOMDP,action::Symbol,state::WorldState)   
    d = ObservationDistribution(pomdp)
    cur_obs = :none
    if colocate_gold(state) && adjacent_wumpus(state)
        cur_obs = :stenchandglitter
    elseif colocate_gold(state)
        cur_obs = :glitter
    elseif adjacent_wumpus(state)
        cur_obs = :stench
    end
    
    d.probabilityVector = vec(fill(0.0, 1, length(d.probabilityVector)))
    for ind=1:length(d.observationVector)
        if d.observationVector[ind]==cur_obs
            d.probabilityVector[ind]=1
        end
    end
    
    
    m_normalize!(d.probabilityVector)
    if sum(d.probabilityVector)==0 print("observation function sum zero") end
    return d
end

POMDPs.iterator(d::ObservationDistribution) = d.observations


function POMDPs.pdf(d::ObservationDistribution,observation::Symbol)
    return d.probabilityVector[findfirst(d.observationVector, observation)]
end

function POMDPs.rand(rng::AbstractRNG, d::ObservationDistribution)
    return sample(d.observationVector, WeightVec(d.probabilityVector))
end

POMDPs.n_states(pomdp::WumpusPOMDP) = pomdp.size_x*pomdp.size_y*pomdp.size_x*pomdp.size_y*pomdp.size_x*pomdp.size_y*2;
POMDPs.n_actions(::WumpusPOMDP) = 5
POMDPs.n_observations(::WumpusPOMDP) = 4;
POMDPs.discount(pomdp::WumpusPOMDP) = pomdp.discount_factor
POMDPs.isterminal(pomdp::WumpusPOMDP, s::WorldState) = s.done

POMDPs.initial_state_distribution(pomdp::WumpusPOMDP,start_agentLoc_x,start_agentLoc_y) = WorldStateDistribution(pomdp,start_agentLoc_x,start_agentLoc_y,false)

POMDPs.initial_state_distribution(pomdp::WumpusPOMDP) = WorldStateDistribution(pomdp,1,1,false)

function POMDPs.state_index(pomdp::WumpusPOMDP,state::WorldState)
    sd = Int(state.done + 1)
    return sub2ind((pomdp.size_x,pomdp.size_y,pomdp.size_x,pomdp.size_y,pomdp.size_x,pomdp.size_y,2),state.agentLoc_x,state.agentLoc_y,state.goldLoc_x,state.goldLoc_y,state.wumpusLoc_x,state.wumpusLoc_y,sd)
end


using SARSOP
#initialize the Wumpus POMDP
pomdp =WumpusPOMDP()

#initialize the solver
solver = SARSOPSolver()

#run the solve function
policy = solve(solver,pomdp)


alphas(policy)

using POMDPToolbox
pomdp = WumpusPOMDP()
init_dist = initial_state_distribution(pomdp,1,1) #starting from 1,1
up = updater(policy) # belief updater for our policy
hist = HistoryRecorder(max_steps=14, rng=MersenneTwister(1)) # history recorder that keeps track of states, observations and beliefs

hist = simulate(hist, pomdp,policy, up, init_dist)

for (s, b, a, r, sp, op) in hist
    println("In state: $s, took action: $a,ended up in state $sp with observation: $op")
    println("$r")
end
println("Total reward: $(discounted_reward(hist))")

pomdp = POMDPFile(pomdp, "model.pomdpx")

import SARSOP.polgraph
import SARSOP.PolicyGraphGenerator
import SARSOP._get_options_list
const EXEC_POLICY_GRAPH_GENERATOR = Pkg.dir("SARSOP", "deps", "polgraph")
graphgen = PolicyGraphGenerator("Grid.dot")
function polgraph(graphgen::PolicyGraphGenerator, pomdp::SARSOPFile, policy::SARSOPPolicy)
    options_list = _get_options_list(graphgen.options)
    run(`$EXEC_POLICY_GRAPH_GENERATOR $(pomdp.filename) --policy-file $(policy.filename) $options_list`)
end
polgraph(graphgen, pomdp, policy)
