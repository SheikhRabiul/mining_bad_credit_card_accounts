DROP TABLE IF EXISTS "config";
CREATE TABLE "config" ("avg_tran_plus_std" DOUBLE, "bad_account_threshold" DOUBLE, "lambda" DOUBLE);
INSERT INTO "config" VALUES(14,60,50);
DROP TABLE IF EXISTS "olap";
CREATE TABLE "olap" ("account" INTEGER DEFAULT (null) ,"statusofexistingcheckingaccount" VARCHAR," credithistory" VARCHAR,"savingaccountorbond" VARCHAR,"installmentrateinpercentangeofdisposablleincome" DOUBLE,"otherdebtorsorguarantors" VARCHAR,"presentresidencesince" INTEGER," otherinstallmentplans" VARCHAR,"job" VARCHAR,"numberofpeoplebeingliabletoprovidemaintenancefor" INTEGER,"telephone" INTEGER,"foreignworker" INTEGER,"status" INTEGER);
DROP TABLE IF EXISTS "oltp";
CREATE TABLE "oltp" (
"index" INTEGER,
  "account" INTEGER,
  "amount" INTEGER,
  "location" TEXT,
  "category" INTEGER,
  "date" TEXT,
  "tid" INTEGER
);
INSERT INTO "oltp" VALUES(0,1,200,'VA',2,'2017-07-27 03:0:0',1);
INSERT INTO "oltp" VALUES(1,2,300,'VA',3,NULL,4);
DROP TABLE IF EXISTS "std_rules";
CREATE TABLE "std_rules" ("id" INTEGER, "rule" VARCHAR, "applicable_to_tran_type" VARCHAR);
INSERT INTO "std_rules" VALUES(1,'Transaction location is near user physical location','expenditure');
INSERT INTO "std_rules" VALUES(2,'Payment within due date','payment');
CREATE INDEX "ix_oltp_index"ON "oltp" ("index");
