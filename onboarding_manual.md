## How to

### root/

```bash
cd .ssh
# 본인 이메일과 본인 성 또는 이름으로 변경
ssh-keygen -t ed25519 -C "bob@example.com" -f /root/.ssh/id_ed25519_bob
# 복사
cat ~/.ssh/id_ed25519_bob.pub

```


깃허브의 Settings -> SSH and GPC keys 에 들어가 New SSH key에 복사한 키를 추가합니다.  

```bash
touch ~/.ssh/config
chmod 600 ~/.ssh/config
```

아래 예시와 같이 config 파일에 본인의 이름을 반영하여 추가합니다.  

```text
Host github-rhee
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_rhee

Host github-bob
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_bob
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
git clone git@github-bob:boostcampaitech8/pro-nlp-mrc-nlp-14.git
cd pro-nlp-mrc-nlp-14

# 본인 깃허브 닉네임과 이메일로 변경해주세요!
git config --local user.name "Your Name"
git config --local user.email "you@example.com"
uv sync
```

이후 커밋할때

### 오류가 나거나 궁금한 사항 있으면 rheefine에게 DM 주세요!!
### 커밋 메세지와 브랜치 명 등은 Confluence의 Convention을 참고해주세요!!
