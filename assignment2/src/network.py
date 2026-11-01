# [수정 가능] ① 모델 구조 ② 보상함수 ③ 수집 행동 선택
# 클래스·함수의 이름과 입출력 규약은 유지하세요. 고정 보조 코드는 파일 하단에 있습니다.
# 모델 입력 (B,96,96,3), 출력 유한한 행동 점수 (B,9). 행동 9개의 순서·의미는 고정입니다.
# 추론·서버 채점은 점수가 가장 높은 행동을 선택합니다.
# 학습은 제공된 Double DQN 코드로 고정합니다.
# 학습 후 저장되는 .network.py와 .pth 두 파일을 제출합니다.
import torch
from pathlib import Path

# ① 모델: 영상·센서 입력으로 행동 9개의 Q값을 계산합니다.
# [고정 규약] ClassificationNetwork(device=None), forward(observation)의 입출력.
class ClassificationNetwork(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()

        device = _setup_model(self, device)

        # 베이스라인과 같은 층. 가중치를 이어 받아 47초 완주 정책에서 미세 조정한다.
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=5, stride=4),
            torch.nn.ReLU(),
            MPSCompatibleAdaptiveAvgPool2d((4, 4)),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(8 * 4 * 4 + 7, self.n_classes),
        )
        self.throttle_head = torch.nn.Linear(1, 2)
        self._load_corner_weights()
        self.to(device)

    def _load_corner_weights(self):
        # 43초 모델의 조향 가중치만 가져온다. 가속 억제 식(-20×속도)은 가져오지 않는다.
        models = Path(__file__).resolve().parents[1] / 'models'
        for name in ('20261027_144131.pth', '20261027_174602.pth', '20261030_171427.pth', 'baseline.pth'):
            path = models / name
            if not path.is_file():
                continue
            payload = torch.load(path, map_location='cpu', weights_only=True)
            state = payload.get('state_dict', payload) if isinstance(payload, dict) else payload
            own = self.state_dict()
            copied = {key: value for key, value in state.items()
                      if key in own and tuple(value.shape) == tuple(own[key].shape) and key.startswith(('conv.', 'fc.'))}
            if copied:
                self.load_state_dict(copied, strict=False)
                return

    def forward(self, observation):
        # 조향은 학습된 특징, 속도 조절은 그 조향이 버티던 40초 완주 규칙.
        # 직진이고 계기판 속도가 1 미만이면 가속, 좌우인데 속도가 0.4 이상이면 브레이크.

        batch_size = observation.shape[0]

        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(
            observation.detach(), batch_size)

        x = observation / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.reshape(batch_size, -1)
        x = torch.cat([x, speed, abs_sensors, steering, gyroscope], dim=1)
        raw = self.fc(x)
        steering_values = torch.stack([raw[:, [0, 1, 2]].max(1).values,
                                       raw[:, [3, 5, 7]].max(1).values,
                                       raw[:, [4, 6, 8]].max(1).values], dim=1)
        steer_index = [0, 0, 0, 1, 2, 1, 2, 1, 2]
        values = steering_values[:, steer_index].clone()
        straight = ((steering_values[:, 0] >= steering_values[:, 1])
                    & (steering_values[:, 0] >= steering_values[:, 2])).float()
        left = ((steering_values[:, 1] > steering_values[:, 0])
                & (steering_values[:, 1] >= steering_values[:, 2])).float()
        right = ((steering_values[:, 2] > steering_values[:, 0])
                 & (steering_values[:, 2] > steering_values[:, 1])).float()
        hud = speed.squeeze(1)
        values[:, 1] = values[:, 1] + 20 * (hud < 1.0).float() * straight
        values[:, 7] = values[:, 7] + 20 * (hud >= 0.4).float() * left
        values[:, 8] = values[:, 8] + 20 * (hud >= 0.4).float() * right
        return values + .01 * raw

    def actions_to_classes(self, actions):
        return _fixed_actions_to_classes(self, actions)

    def scores_to_action(self, scores):
        return _fixed_scores_to_action(self, scores)

    def extract_sensor_values(self, observation, batch_size):
        return _fixed_extract_sensor_values(self, observation, batch_size)

# ② 보상: 아래 계수 또는 calculate_reward()를 수정한 뒤 데이터를 다시 수집합니다.
# 랩타임이 목표다. 도로 위 속도에 보상하고, 잔디에 오래 머물면 감점한다.
REWARD_SETTINGS = dict(new_road=1.0, center=0.3, grass=1.5, time=0.02, finish=10.0, speed=0.03)

# [고정 규약] calculate_reward(context)는 유한한 숫자 하나를 반환합니다.
def calculate_reward(context):
    # 도로 위에 있는 동안의 속도가 랩타임을 줄인다. 한가운데 유지는 약한 가산만 둔다.
    # context에는 speed, dt, elapsed, aborted도 제공됩니다.

    center = context['center_alignment']
    reward = context['new_road_segments'] * (
        REWARD_SETTINGS['new_road'] + REWARD_SETTINGS['center'] * center)
    if context['on_road']:
        reward += REWARD_SETTINGS['speed'] * context['speed']
    else:
        reward -= REWARD_SETTINGS['grass']
    reward -= REWARD_SETTINGS['time']
    if context['finished']:
        reward += REWARD_SETTINGS['finish']
    return reward


# ③ 수집: 10바퀴 모두 40초에 완주했던 규칙만 사용한다.
# 속도를 그보다 올리면 이 조향은 코너를 벗어난다.
def collection_action(model, observation, device, epsilon, rng):
    # [고정 규약] 위 함수 이름·인자 유지, 반환값은 행동 번호 0~8의 정수.
    with torch.inference_mode():
        scores = model(torch.as_tensor(observation[None].copy(), device=device, dtype=torch.float32))
        if scores.shape != (1, 9) or not torch.isfinite(scores).all():
            raise ValueError('모델 출력은 유한한 (1, 9) 행동 점수여야 합니다.')
        action = int(scores.argmax(1).item())
    if epsilon <= 0:
        return action
    speed = float(observation[84:94, 12, 0].sum()) / 255.0 / 5.0
    if action == 0 and speed < 1.0:
        return 1
    if action == 3 and speed >= 0.4:
        return 7
    if action == 4 and speed >= 0.4:
        return 8
    return action


# ===========================================
# ===== 고정 보조 코드: 아래는 수정하지 않습니다 =====
# ===========================================

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def _setup_model(self, device):
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    self.device = device

    self.n_classes = 9
    # [고정] 행동 순서·제어값: 중립, 가속, 제동, 좌, 우, 좌+가속, 우+가속, 좌+제동, 우+제동.
    self.action_classes = [
        (0.0,  0.0, 0.0),
        (0.0,  0.5, 0.0),
        (0.0,  0.0, 0.8),
        (-1.0, 0.0, 0.0),
        (1.0,  0.0, 0.0),
        (-1.0, 0.5, 0.0),
        (1.0,  0.5, 0.0),
        (-1.0, 0.0, 0.8),
        (1.0,  0.0, 0.8),
    ]

    return device

def _fixed_actions_to_classes(self, actions):

    class_indices = []
    for action in actions:
        steer = action[0].item()
        gas   = action[1].item()
        brake = action[2].item()

        is_left  = steer < -0.1
        is_right = steer > 0.1
        is_gas   = gas   > 0.1
        is_brake = brake > 0.1

        if is_left and is_gas:
            cls = 5
        elif is_right and is_gas:
            cls = 6
        elif is_left and is_brake:
            cls = 7
        elif is_right and is_brake:
            cls = 8
        elif is_left:
            cls = 3
        elif is_right:
            cls = 4
        elif is_gas:
            cls = 1
        elif is_brake:
            cls = 2
        else:
            cls = 0

        class_indices.append(torch.tensor([cls], dtype=torch.long))
    return class_indices


def _fixed_scores_to_action(self, scores):

    cls = torch.argmax(scores, dim=1).item()
    return self.action_classes[cls]


def _fixed_extract_sensor_values(self, observation, batch_size):

    speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
    speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

    abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
    abs_sensors = abs_crop.sum(dim=1) / 255 / 5

    steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
    steer_crop[:, :10] *= -1
    steering = steer_crop.sum(dim=1, keepdim=True)

    gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
    gyro_crop[:, :14] *= -1
    gyroscope = gyro_crop.sum(dim=1, keepdim=True)

    return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


# MPS 호환 보조 계층: 풀링 계산은 유지합니다.
class MPSCompatibleAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    def forward(self, x):
        if x.device.type != 'mps':
            return super().forward(x)
        height, width = x.shape[-2:]
        size = self.output_size
        out_h, out_w = (size, size) if isinstance(size, int) else size
        out_h = height if out_h is None else out_h
        out_w = width if out_w is None else out_w
        if out_h <= 0 or out_w <= 0 or height == 0 or width == 0:
            return super().forward(x)
        if height % out_h == 0 and width % out_w == 0:
            return super().forward(x)

        rows = []
        for i in range(out_h):
            start_h = i * height // out_h
            end_h = ((i + 1) * height + out_h - 1) // out_h
            cells = []
            for j in range(out_w):
                start_w = j * width // out_w
                end_w = ((j + 1) * width + out_w - 1) // out_w
                cells.append(x[..., start_h:end_h, start_w:end_w].mean(dim=(-2, -1)))
            rows.append(torch.stack(cells, dim=-1))
        return torch.stack(rows, dim=-2)

