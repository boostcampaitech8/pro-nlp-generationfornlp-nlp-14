## How to

### root/

```bash
cd .ssh
# 본인 이메일과 본인 성 또는 이름으로 변경
ssh-keygen -t ed25519 -C "bob@example.com" -f /root/.ssh/id_ed25519_bob
# 엔터 두번 입력

# cat 명령어를 통해 public key 조회
cat /root/.ssh/id_ed25519_bob.pub
# 조회한 key를 복사해주세요
```

깃허브의 Settings -> SSH and GPC keys 에 들어가 New SSH key에 복사한 키를 추가합니다.  

```bash
touch /root/.ssh/config
chmod 600 /root/.ssh/config
```

아래 예시와 같이 config 파일에 본인의 이름을 반영하여 추가합니다.  

```text
Host github-rhee
    HostName github.com
    User git
    IdentityFile /root/.ssh/id_ed25519_rhee

Host github-bob
    HostName github.com
    User git
    IdentityFile /root/.ssh/id_ed25519_bob
```


### /data/ephemeral/home

```bash
apt update
# git 설치
apt install git
# make 설치
apt install make
```

```bash
# uv 설치
python3 -m pip install --upgrade uv
# 본인이 작업할 디렉토리 만들기
mkdir bob
cd bob

# github-bob 대신 ssh 설정시 상용한 본인 이름
git clone git@github-bob:boostcampaitech8/pro-nlp-generationfornlp-nlp-14.git csat
# Are you sure you want to continue connecting (yes/no/[fingerprint])? 
# 위와 같은 입력 창이 나오면 yes
cd csat

# 본인 깃허브 닉네임과 이메일로 변경해주세요!
git config --local user.name "Your Name"
git config --local user.email "you@example.com"
uv sync
```
# 커밋 메세지와 브랜치명 등은 Confluence의 Convention을 참고해 반드시 맞춰주세요!!
