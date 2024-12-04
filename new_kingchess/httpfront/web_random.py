
from agent import random_agent
from httpfront.app import get_web_app
from net.mcts_alphazreo import MCTSPlayer
from net.policy_value_net_pytorch import PolicyValueNet

myagent = random_agent.Random_agent()
policy_value_net_current = PolicyValueNet(model_file='E:/new_model_4_19/get_muc_model_5_28/current.pt')
mcts_current = MCTSPlayer(policy_value_net_current.policy_value_fn, c_puct=5, n_playout=1200)
web_app, app= get_web_app({'random': myagent, 'mcts': mcts_current})
web_app.run(app, host='0.0.0.0', port=5000, debug=True)
# # myagent = random_agent.Random_agent()
# web_app = get_web_app_1({'random': myagent})
# web_app.run(host='0.0.0.0', port=8888)