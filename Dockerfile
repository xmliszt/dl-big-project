FROM ubuntu:latest
RUN apt-get -y update
RUN apt-get install python3 python-pip -y
WORKDIR /model
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /
COPY packege*.json .
RUN npm install
WORKDIR /client
COPY package*.json .
RUN npm install
WORKDIR /
COPY . .
ENV NODE_ENV=production
CMD ["npm", "start"]