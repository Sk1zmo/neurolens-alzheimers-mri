import Link from "next/link";

export const metadata = { title: "How it works" };

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card p-5 sm:p-6">
      <h2 className="text-base font-semibold tracking-tight">{title}</h2>
      <div className="mt-3 space-y-3 text-[0.875rem] leading-relaxed text-[var(--text-secondary)]">
        {children}
      </div>
    </section>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="flex gap-3.5">
      <span
        className="tnum mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[0.75rem] font-semibold"
        style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
      >
        {n}
      </span>
      <div>
        <p className="text-[0.875rem] font-medium text-[var(--text-primary)]">{title}</p>
        <p className="mt-1 text-[0.8125rem] leading-relaxed text-[var(--text-secondary)]">
          {children}
        </p>
      </div>
    </li>
  );
}

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6 sm:py-10">
      <header className="mb-7">
        <h1 className="text-[1.75rem] font-semibold tracking-tight">How it works</h1>
        <p className="mt-3 text-[0.9375rem] leading-relaxed text-[var(--text-secondary)]">
          What the model is, what the data actually supports, and what happens to
          a scan you upload.
        </p>
      </header>

      <div className="space-y-5">
        <section
          className="rounded-xl border p-5"
          style={{
            borderColor: "color-mix(in srgb, var(--critical) 40%, transparent)",
            background: "color-mix(in srgb, var(--critical) 8%, transparent)",
          }}
        >
          <h2 className="text-base font-semibold tracking-tight">
            This is not a diagnostic tool
          </h2>
          <p className="mt-2 text-[0.875rem] leading-relaxed text-[var(--text-secondary)]">
            NeuroLens is a research and teaching demonstration. It is not a
            medical device, has no regulatory clearance, and has never been
            validated on a clinical population. It cannot diagnose Alzheimer&apos;s
            disease in anyone, and no output from it should influence a medical
            decision. If you are worried about cognitive symptoms — yours or
            someone else&apos;s — talk to a doctor.
          </p>
        </section>

        <Section title="MRI, not CT">
          <p>
            The model was trained on the Kaggle{" "}
            <em>augmented-alzheimer-mri-dataset</em>, which is axial T1-weighted
            brain <strong className="font-medium text-[var(--text-primary)]">MRI</strong>.
            CT and MRI produce fundamentally different tissue contrast, so a CT
            slice falls outside everything this model has seen. It will still
            return a stage and a confident-looking percentage — that output is
            meaningless.
          </p>
          <p>
            Two guards exist for this. A structural check tests whether the
            upload even looks like an axial brain slice (near-greyscale, dark
            surround, one centred mass, real internal texture). Separately, a
            free-energy score flags inputs that sit beyond the 99th percentile of
            the held-out test distribution. Both surface as warnings on the
            result rather than silently changing the answer.
          </p>
        </Section>

        <Section title="The honest version of the accuracy number">
          <p>
            The obvious way to use this dataset — train on the 33,984 augmented
            images, test on the 6,400 originals — is the recipe most published
            notebooks follow, and it reports something like 99%. It is also
            wrong: the augmented images are derived from the originals, so that
            protocol tests on data the model has already effectively memorised.
          </p>
          <p>
            NeuroLens splits differently. The test set is carved out of the
            original images only. Then every augmented image is matched to its
            single most similar original in embedding space, and any augmented
            image whose source landed in validation or test is thrown away before
            training. That removed 11,878 of 33,984 augmented images — 35.0%,
            almost exactly the 35% of originals held out, which is the sanity
            check that the matching worked.
          </p>
          <p>
            One problem no split can fix: the dataset has no subject
            identifiers. Multiple slices from the same brain appear throughout,
            and there is no way to keep one patient wholly on one side of the
            split. Real-world accuracy on a genuinely new patient is lower than
            the figure on the{" "}
            <Link href="/model" className="text-[var(--accent)] hover:underline">
              model card
            </Link>
            . That page states the number anyway, with this caveat attached,
            because a metric without its caveat is worse than no metric.
          </p>
        </Section>

        <Section title="How a prediction is produced">
          <ol className="space-y-4">
            <Step n={1} title="Preprocess">
              The slice is resized to 224×224 and normalised with ImageNet
              statistics — byte-for-byte the same transform used in training.
            </Step>
            <Step n={2} title="Classify">
              An EfficientNet-B0 backbone, fine-tuned on the leak-filtered
              training set, produces four logits. Serving runs on ONNX Runtime,
              not PyTorch, because a PyTorch wheel alone exceeds the serverless
              bundle limit.
            </Step>
            <Step n={3} title="Calibrate">
              Raw softmax from a fine-tuned CNN is over-confident. A single
              temperature parameter, fitted on the validation split, rescales the
              distribution so a stated 80% corresponds to roughly 80% observed
              accuracy.
            </Step>
            <Step n={4} title="Explain">
              The classifier head is deliberately global-average-pool followed by
              one linear layer. That makes the classic class activation map exact
              and gradient-free — it is a 1×1 convolution, folded into the ONNX
              graph at export time. The heatmap costs one forward pass and no
              autograd.
            </Step>
          </ol>
          <p className="pt-1">
            A class activation map shows where the model looked, not that the
            highlighted tissue is abnormal. Reading it as a lesion marker is the
            single most common way saliency maps get over-interpreted.
          </p>
        </Section>

        <Section title="What happens to a scan you upload">
          <p>
            Nothing is stored unless you press <em>Save to history</em>. Analysis
            itself is stateless: the image is posted to the inference function,
            classified, and dropped.
          </p>
          <p>When you do save, the scan is written twice, on purpose:</p>
          <ul className="ml-4 list-disc space-y-2 marker:text-[var(--text-muted)]">
            <li>
              <strong className="font-medium text-[var(--text-primary)]">
                Your copy
              </strong>{" "}
              — appears in{" "}
              <Link href="/history" className="text-[var(--accent)] hover:underline">
                history
              </Link>
              , tied to a random id in this browser&apos;s localStorage. There are
              no accounts and nothing links a scan to a person. Clearing site data
              ends that identity for good.
            </li>
            <li>
              <strong className="font-medium text-[var(--text-primary)]">
                The retraining copy
              </strong>{" "}
              — only if you leave the contribution checkbox ticked. It sits in a
              queue and becomes training data only after a human assigns it a
              verified label in the review console. The model&apos;s own guess is
              never used as ground truth; training on your own predictions just
              amplifies whatever the model already believes.
            </li>
          </ul>
          <p>
            Deleting from history removes your copy and the stored image. If the
            contribution had not yet been reviewed, it is withdrawn too. If it had
            already been reviewed and used in a training run, the labelled record
            stays — otherwise the corpus behind a completed run would change
            underneath it and its before/after comparison would stop meaning
            anything.
          </p>
          <p>
            Please do not upload identifiable patient data. This is a
            demonstration project with no clinical data-handling guarantees, no
            BAA, and no compliance posture.
          </p>
        </Section>

        <Section title="Stack">
          <ul className="ml-4 list-disc space-y-1.5 marker:text-[var(--text-muted)]">
            <li>Next.js App Router frontend, deployed on Vercel</li>
            <li>
              Inference in a Vercel Python function: ONNX Runtime + NumPy +
              Pillow, no PyTorch and no OpenCV
            </li>
            <li>
              Supabase Postgres for scan records and the review queue, Supabase
              Storage (private bucket, signed URLs) for the images
            </li>
            <li>
              Training in PyTorch: EfficientNet-B0, two-stage fine-tune,
              class-weighted loss with a balanced sampler, one-cycle schedule
            </li>
          </ul>
        </Section>

        <Section title="Dataset credit">
          <p>
            Kaggle:{" "}
            <a
              href="https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset"
              target="_blank"
              rel="noreferrer noopener"
              className="text-[var(--accent)] hover:underline"
            >
              uraninjo/augmented-alzheimer-mri-dataset
            </a>
            . Four classes: NonDemented, VeryMildDemented, MildDemented,
            ModerateDemented. Note that ModerateDemented has only 64 original
            slices in the entire dataset, which is why its held-out metrics carry
            wide error bars.
          </p>
        </Section>
      </div>
    </div>
  );
}
